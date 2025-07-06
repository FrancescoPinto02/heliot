import pandas as pd
import requests
import json
import time
from tqdm import tqdm
import argparse

# Process the HELIOT service response
def process_response(response):
    result_text = ""
    data = {}
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')

            if "data: {\"message\":" in decoded_line:
                j = decoded_line.rfind("\"}")
                decoded_line = decoded_line[5:]
                mess = json.loads(decoded_line)
                decoded_line = mess['message']
            
            if not "data: {\"input\":" in decoded_line and not "data: {'input':" in decoded_line:
                result_text += decoded_line
            else:
                decoded_line = decoded_line[5:]
                data = json.loads(decoded_line) 
                print("QUI", data)               

    try:
        result_json = json.loads(result_text)
        return result_json, data
    except:
        return None, data

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Perform the full experiment on the synthetic dataset.")
    
    # Add arguments for the files
    parser.add_argument("--file", default="patients_synthetic.xlsx", help="Path to the synthetic dataset.")
    parser.add_argument("--out", default="results_full_synth.xlsx", help="Output file.")
    args = parser.parse_args()

    # 1. Read the patients_synthetic dataset
    df = pd.read_excel(args.file, dtype={'drug_code': str, 'leaflet': str, 'patient_id': str})
    
    # List for the results 
    results = []
    
    # URL of the HELIO service
    url = "http://localhost:8000/api/allergy_check_enhanced"
    
    # 2. Process each row
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing results..."):

        
        payload = {
            "patient_id": row['patient_id'],
            "drug_code": row['drug_code'],
            "clinical_notes": row['clinical_note'],
            "store": False  
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, stream=False)
            end = time.time()
        except Exception as ec:
            print(ec)
            time.sleep(20)    
            start_time = time.time()
            response = requests.post(url, json=payload, stream=False)
            end = time.time()

        try:            
            if response.status_code == 200:
                result_json, data = process_response(response)
                if result_json:
                    # Add all the synthetic_patients dataset fields
                    result_row = row.to_dict()
                    
                    # Add the result fields
                    result_row['timing'] = end - start_time
                    result_row['response'] = result_json.get('a', '')
                    result_row['classification_resp'] = result_json.get('r', '')
                    result_row['reaction_resp'] = result_json.get('rt', '')
                    result_row['confidence'] = data.get('confidence',0.0)
                    result_row['analysis_confidence'] = data.get('a_confidence',0.0)
                    result_row['case_confidence'] = data.get('r_confidence',0.0)
                    result_row['reaction_confidence'] = data.get('rt_confidence',0.0)
                    
                    if result_row['analysis_confidence'] == 0.0:
                        print("=============>", data)

                    results.append(result_row)
                else:
                    print("VUOTO", result_json)
            else:
                print(f"Error in request for patient_id {row['patient_id']}: {response.status_code}")
                raise
            #if response.status_code == 200:
            #    break
        except Exception as e:
            print(f"Error for patient_id {row['patient_id']}: {str(e)}")
    
    # 3. Create a new DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Save the Excel
    output_filename = args.out #'results_full_synth.xlsx'
    results_df.to_excel(output_filename, index=False)
    print(f"Risultati salvati in {output_filename}")

if __name__ == "__main__":
    main()