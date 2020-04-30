import os
import requests
import shutil
import gzip
import pyarrow as pa
from dask.distributed import Client
from dask import delayed
import cudf

from dash_rapids_mortgage.utils import scheduler_url

print(f"Connecting to cluster at {scheduler_url} ... ", end='')
client = Client(scheduler_url)
print("done")


def load_dataset(path):
    """
    Args:
        path: Path to arrow file containing mortgage dataset

    Returns:
        pandas DataFrame
    """
    # Load dataset as pyarrow table
    reader = pa.RecordBatchStreamReader(path)
    pa_table = reader.read_all()

    # Convert to pandas DataFrame
    pd_df = pa_table.to_pandas()

    # # # Subset to 40M for debugging
    # pd_df = pd_df.head(40000000)

    # Convert zip to int16
    pd_df['zip'] = pd_df['zip'].astype('int16')

    # drop extra columns
    pd_df.drop(['loan_id', 'seller_name'], axis=1, inplace=True)

    return pd_df


if __name__ == '__main__':
    # development entry point
    # Look for dataset
    dataset_url = 'https://s3.us-east-2.amazonaws.com/rapidsai-data/viz-data/146M_predictions_v2.arrow.gz'

    data_path = "./data/146M_predictions_v2.arrow"
    if not os.path.exists(data_path):
        print(f"Mortgage dataset not found at ./data/146M_predictions_v2.arrow.\n"
              f"Downloading from {dataset_url}")
        # Download dataset to data directory
        os.makedirs('./data', exist_ok=True)
        data_gz_path = data_path + '.gz'
        with requests.get(dataset_url, stream=True) as r:
            r.raise_for_status()
            with open(data_gz_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print("Decompressing...")
        with gzip.open(data_gz_path, 'rb') as f_in:
            with open(data_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        print("Deleting compressed file...")
        os.remove(data_gz_path)

        print('done!')
    else:
        print(f"Found dataset at {data_path}")


    def load_and_publish_dataset():
        # pandas DataFrame
        pd_df_d = delayed(load_dataset)(data_path).persist()

        # cudf DataFrame
        c_df_d = delayed(cudf.DataFrame.from_pandas)(pd_df_d).persist()

        # Clear all published datasets
        for k in client.list_datasets():
            client.unpublish_dataset(k)

        # Replicate datasets to all workers (GPUs) on machine
        client.replicate(pd_df_d)
        client.replicate(c_df_d)

        # Publish datasets to the cluster
        client.publish_dataset(pd_df_d=pd_df_d)
        client.publish_dataset(c_df_d=c_df_d)

    load_and_publish_dataset()
