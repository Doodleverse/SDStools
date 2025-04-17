# adds geojson handler, additional csv input and epsg code
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import logging

# Basic example of running the script:

# python csv_converter_v2.py -i 'C:\development\doodleverse\SDS_tools\csv_converter\CSV_sample_data\LB_ROI1_v3_transect_time_series_tidally_corrected (1).csv' -o 'C:\development\doodleverse\SDS_tools\csv_converter\ouput_CSVs\LB_ROI1_v3_transect_time_series_tidally_corrected.csv'
# -i : input file path 
# -o : output file path
# --geojson : path to geojson file containing transects (optional)
# --ref_csv : path to reference CSV file containing additional columns (optional)
# --crs : coordinate reference system code to convert geojson to (optional, default is epsg:4326)

def configure_logging():
    """Configure and return a logger for the script."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert a CSV file into a standardized CSV format with optional GeoJSON and reference CSV.")
    parser.add_argument("-i","--input_file", help="Path to the input CSV file")
    parser.add_argument("-o","--output_file", help="Path to the output (standardized) CSV file")
    parser.add_argument("--geojson", help="Optional: GeoJSON file containing transects", default=None)
    parser.add_argument("--ref_csv", help="Optional: Add missing columns from reference CSV file containing additional columns. Must have column 'transect_id'", default=None)
    parser.add_argument("--crs", help="Optional: CRS code (e.g., epsg:4326)", default="epsg:4326")
    return parser.parse_args()

def load_csv(filepath, logger):
    """Load a CSV file and drop any unnamed columns."""
    logger.info(f"Loading CSV from: {filepath}")
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def add_missing_columns(df, standard_cols, logger):
    """Add any missing standardized columns to the DataFrame."""
    for col in standard_cols:
        if col not in df.columns:
            df[col] = np.nan
            logger.info(f"Added missing column: {col}")
    return df

def merge_reference_csv(df, ref_csv_filepath, standard_cols, logger):
    """Merge the DataFrame with a reference CSV and fill missing values for standard columns."""
    logger.info(f"Loading reference CSV from: {ref_csv_filepath}")
    ref_df = pd.read_csv(ref_csv_filepath)
    df = df.merge(ref_df, on='transect_id', how='left', suffixes=('', '_ref'))
    logger.info("Merged input CSV with reference CSV on transect_id")
    for col in standard_cols:
        ref_col = col + "_ref"
        if ref_col in df.columns:
            missing_before = df[col].isna().sum()
            df[col] = df[col].fillna(df[ref_col])
            missing_after = df[col].isna().sum()
            logger.info(f"Filled {missing_before - missing_after} missing values in '{col}' using reference CSV")
            df.drop(columns=[ref_col], inplace=True)
    return df

def extract_endpoint_from_geometry(geom, logger):
    """Extract the last coordinate from a geometry if it is a LineString or MultiLineString."""
    if geom is None:
        return None
    try:
        if geom.geom_type == "LineString":
            return list(geom.coords)[-1]
        elif geom.geom_type == "MultiLineString":
            last_line = list(geom.geoms)[-1]
            return list(last_line.coords)[-1]
        else:
            logger.warning(f"Unsupported geometry type '{geom.geom_type}'")
            return None
    except Exception as e:
        logger.error(f"Error processing geometry: {e}")
        return None

def process_geojson(geojson_filepath, crs, logger):
    """Read a GeoJSON file and return a mapping from transect_id to its endpoint coordinates."""
    try:
        import geopandas as gpd
    except ImportError:
        logger.error("geopandas module is required for geojson processing. Please install geopandas.")
        raise

    logger.info(f"Reading GeoJSON file from: {geojson_filepath}")
    gdf = gpd.read_file(geojson_filepath)
    
    if gdf.crs != crs:
        logger.info(f"Reprojecting GeoJSON data from {gdf.crs} to {crs}")
        gdf = gdf.to_crs(crs)
    
    if 'transect_id' not in gdf.columns:
        logger.error("GeoJSON file does not contain a 'transect_id' field.")
        return {}
    
    endpoints = {}
    for idx, row in gdf.iterrows():
        transect_id = row['transect_id']
        coord = extract_endpoint_from_geometry(row.geometry, logger)
        if coord:
            endpoints[transect_id] = coord
            logger.info(f"Extracted endpoint for transect_id {transect_id}: {coord}")
    return endpoints

def update_transect_endpoints(df, endpoints, logger):
    """Update the transect_end_x and transect_end_y in the DataFrame based on endpoints mapping."""
    match_found = False
    for i, row in df.iterrows():
        t_id = row['transect_id']
        if t_id in endpoints:
            x, y = endpoints[t_id]
            df.at[i, 'transect_end_x'] = x
            df.at[i, 'transect_end_y'] = y
            match_found = True
            logger.info(f"Updated row {i} for transect_id {t_id}: transect_end_x={x}, transect_end_y={y}")
    if not match_found:
        logger.warning("No matching transect_ids found between input CSV and GeoJSON file")
    return df

def reorder_columns(df, standard_cols, logger):
    """Reorder the DataFrame columns to match the standard order."""
    logger.info("Reordering columns to the standard order")
    return df[standard_cols]

def save_standardized_csv(df, output_filepath, logger):
    """Save the standardized DataFrame to a CSV file."""
    df.to_csv(output_filepath, index=False)
    logger.info(f"Standardized CSV saved to: {output_filepath}")

def standardize_csv(input_filepath, output_filepath, geojson_filepath=None, ref_csv_filepath=None, crs='epsg:4326'):
    logger = configure_logging()
    logger.info("Starting CSV standardization process")
    
    # Define standardized column order.
    standard_cols = [
        'dates',
        'transect_end_x',
        'transect_end_y',
        'tide',
        'transect_id',
        'cross_distance',
        'x',
        'y',
        'shift_x',
        'shift_y',
        'satname',
        'image_suitability_score',
        'segmentation_suitability_score',
        'kde_score',
        'avg_suitability',
        'total_water_level',
        'slope',
        'reference_elevation'
    ]
    
    # Load and pre-process CSV.
    df = load_csv(input_filepath, logger)
    df = add_missing_columns(df, standard_cols, logger)

    # Merge with reference CSV if provided.
    if ref_csv_filepath:
        df = merge_reference_csv(df, ref_csv_filepath, standard_cols, logger)
    
    # Process GeoJSON transects if provided.
    if geojson_filepath:
        endpoints = process_geojson(geojson_filepath, crs, logger)
        df = update_transect_endpoints(df, endpoints, logger)
    
    # Reorder and save the final DataFrame.
    df = reorder_columns(df, standard_cols, logger)
    save_standardized_csv(df, output_filepath, logger)
    logger.info("CSV standardization process completed")

if __name__ == "__main__":
    args = parse_arguments()
    standardize_csv(
        args.input_file,
        args.output_file,
        geojson_filepath=args.geojson,
        ref_csv_filepath=args.ref_csv,
        crs=args.crs
    )
