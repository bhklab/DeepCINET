import pandas as pd
import yaml

def main(cfg):
    radiomic = pd.read_csv(cfg['radiomic_path'])

    radiomic = radiomic.set_index('PatientID').T
    radiomic.drop(["Study", 'image', 'mask'], inplace=True)
    radiomic.to_csv('radiomic_features.csv', index=False)
    print(cfg['clinic_path'])
    clinic = pd.read_excel(cfg['clinic_path'])

    clinic.rename(columns={"ID": "id", "DFS time": "time", "DFS status (1= event)": "event"}, inplace=True)
    clinic.to_csv('clinical_info.csv')

if __name__ == "__main__":

    with open("config.yml", 'r') as cfg_file:
        cfg = yaml.load(cfg_file)
    main(cfg)
