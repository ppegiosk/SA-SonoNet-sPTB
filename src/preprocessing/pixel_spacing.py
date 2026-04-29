import argparse
import numpy as np
import pandas as pd 
import pydicom
import warnings
warnings.filterwarnings("ignore")

from .utils import remove_gray_image

def get_info_from_dicom_files(csv, img_size=(224,288)):

    df = pd.read_csv(csv)

    df['pixel_spacing'] = pd.Series([])
    df['pixel_rows'] = pd.Series([])
    df['pixel_columns'] = pd.Series([])
    df['physical_delta_x'] = pd.Series([])
    df['physical_delta_y'] = pd.Series([])
    df['px_spacing'] = pd.Series([])
    df['py_spacing'] = pd.Series([])
    df['is_rgb'] = pd.Series([])
    df['date'] = pd.Series([])
    df['device_name'] = pd.Series([])
    df['serial_number'] = pd.Series([])

    for i in range(len(df)):
        # path: path to dicom ultrasound image in the database
        print(i, df['path'][i])

        ds = pydicom.dcmread(df['path'][i])
        dicom_name = ds

        # image_dir_calipers: path to jpg ultrasound image
        image_name = df['image_dir_calipers'][i].split('/')[-1]
        
        df['is_rgb'][i] = remove_gray_image(df['image_dir_calipers'][i])
        dicom = ds
        df['date'][i]  = dicom.StudyDate
        
        try:
            device = dicom.ManufacturerModelName
        except:
            device = None

        try:
            serial_number = dicom.DeviceSerialNumber
        except:
            serial_number = None

        df['device_name'][i] = device
        df['serial_number'][i] = serial_number
        df['pixel_rows'][i] = int(ds.Rows)
        df['pixel_columns'][i] = int(ds.Columns)

        if hasattr(ds, 'PixelSpacing'):
            df['pixel_spacing'][i] = np.unique(np.array((ds.PixelSpacing)))

        if [0x0018, 0x6011] in ds:
            df['physical_delta_x'][i] = ds[0x0018, 0x6011][0][0x0018, 0x602c].value
            df['physical_delta_y'][i] = ds[0x0018, 0x6011][0][0x0018, 0x602e].value
            df['px_spacing'][i] = int(ds.Rows) / img_size[0] * df['physical_delta_x'][i]
            df['py_spacing'][i] = int(ds.Columns) / img_size[1] * df['physical_delta_y'][i]

    updated_csv_name = args.csv.split('.csv')[0] + '_pixel_spacing' + '.csv'
    df.to_csv(f'{updated_csv_name}', index=False)
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Retrieve pixel spacing and other information from dicom files')
    parser.add_argument('--csv', type=str, help='path to csv with input image paths', default='/home/ppar/SA-SonoNet-sPTB/metadata/dataset_preterm.csv')
    args = parser.parse_args()

    df = get_info_from_dicom_files(args.csv, img_size=(224,288))