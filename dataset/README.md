# Dataset Placeholders

The following large dataset files are required for this project but are excluded from the repository due to their size:

- `2019-Oct-Optimized.parquet`
- `2019-Nov-Optimized.parquet`
- `sessions_df_final_project.parquet`

## How to Get the Data

1.  **Download Raw Data**: Download `2019-Oct.csv` and `2019-Nov.csv` from the [eCommerce Events History dataset on Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store).
2.  **Place in Dataset Folder**: Put the CSV files in this `dataset/` directory.
3.  **Run Processing Script**: Execute `python process_data_safe.py` to generate the optimized parquet files.

*Note: The `.parquet` files are listed in `.gitignore` to prevent accidental commits of large files.*
