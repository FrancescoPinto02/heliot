import streamlit as st
import requests
import json
import pandas as pd
from streamlit_searchbox import st_searchbox
import time

# URL for the REST service
url = "http://localhost:8000/api/allergy_check_enhanced"


df_d = pd.read_csv('leaflet_info.csv')

def remove_ending_quote(s):
    if s.endswith('"'):
        return s.rstrip('"')
    return s

def process_response(response):
    result_text = ""
    try:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if "data: {\"message\":" in decoded_line:
                    j = decoded_line.rfind("\"}")
                    decoded_line = decoded_line[5:]
                    mess = json.loads(decoded_line)
                    decoded_line = mess['message']
                
                if not "data: {\"input\":" in decoded_line:
                    result_text += decoded_line
    except:
        pass

    try:
        result_json = json.loads(result_text)
        return result_json
    except:
        return None
    
def search_drugs(searchterm: str) -> list:
    if not searchterm:
        return []
    
    # Converti le colonne in stringhe
    df_d['drug_code'] = df_d['drug_code'].astype(str)
    df_d['drug_name'] = df_d['drug_name'].astype(str)
    
    # Aggiungi regex=False per trattare i caratteri speciali come normali caratteri
    mask = (df_d['drug_code'].str.contains(searchterm, case=False, na=False, regex=False) |
           df_d['drug_name'].str.contains(searchterm, case=False, na=False, regex=False))
    
    # Ordina per drug_name prima di creare la lista dei risultati
    results = df_d[mask].sort_values('drug_name').apply(
        lambda x: f"{x['drug_name']} ({x['drug_code']})", 
        axis=1
    ).tolist()
    
    return results[:30]

def search_drugs_(searchterm: str) -> list:
    if not searchterm:
        return []
    
    # Converti le colonne in stringhe
    df_d['drug_code'] = df_d['drug_code'].astype(str)
    df_d['drug_name'] = df_d['drug_name'].astype(str)
    
    # Ora puoi usare str.contains
    mask = (df_d['drug_code'].str.contains(searchterm, case=False, na=False) |
           df_d['drug_name'].str.contains(searchterm, case=False, na=False))
    
    results = df_d[mask].apply(lambda x: f"{x['drug_name']} ({x['drug_code']})", axis=1).tolist()
    
    return results[:30]


# Title of the Web App
st.title("Heliot DSS POC")

# Create tabs
tab1, tab2 = st.tabs(["Try Heliot", "Load csv"])

with tab1:
    # Input for drug code and allergy
    patient_id = st.text_input("Fill in the patient ID:", "043955018")

    selected_drug = st_searchbox(
        search_drugs,
        placeholder="Search drug by name or code...",
        key="drug_search"
    )

    drug_code = None
    if selected_drug:
        drug_code = selected_drug.split('(')[-1].strip(')')
        if len(drug_code) <9:
            drug_code = drug_code.zfill(9)
            #drug_code = "0"+drug_code
        st.write(f"Selected code: {drug_code}")

    #drug_code = st.text_input("Fill in the drug code:", "043955018")
    allergy = st.text_input("Fill in the medical narrative related to the patient (empty means no allergy or intollerance):", "No known allergies or intolerances")

    store = st.checkbox('Store medical narrative', value=True)

    # Text Area to show the results
    text_area = st.empty()

    if drug_code and st.button("Submit"):
        payload = {"patient_id": patient_id, "drug_code": drug_code, "clinical_notes": allergy, "store": store}

        with st.spinner("Loading..."):
            response = requests.post(url, json=payload, stream=True)

            if response.status_code == 200:
                result_text = ""
                tk = {}
                result_container = st.empty()
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if "data: {\"message\":" in decoded_line:
                            j = decoded_line.rfind("\"}")
                            decoded_line = decoded_line[5:]
                            mess = json.loads(decoded_line)
                            decoded_line = mess['message']
                        
                        if not "data: {\"input\":" in decoded_line and not "data: {\'input\':" in decoded_line:
                            result_text += decoded_line 
                        
                        #print("QUI", result_text)

                        result_text = result_text.replace('{"a":"','').replace('","r":"','\n\n').replace('","rt":"','\n\n').replace("}","")
                        result_text = result_text.replace('", "rt":"','\n\n').replace('", "r":"','\n\n').replace("}",'')
                        result_container.write(result_text)
                        #text_area.text_area("Result", result_text, height=400, key="result_textarea")
                        if "data: {\"input\":" in decoded_line or "data: {\'input\':" in decoded_line:
                            tokens = decoded_line[5:]
                            print(tokens)
                            tk = json.loads(tokens)
                            print("\nTOKENS", tk['input'], tk['output'], tk['total'])

                if result_text:
                    result_text = remove_ending_quote(result_text)
                    #result_container.write(result_text)
                    # Determina l'icona in base alla gravità
                    severity_icon = ""
                    last_part = result_text[-200:].lower()  # Considera gli ultimi 200 caratteri
                    
                    if "life-threatening" in last_part and "non life-threatening" not in last_part:
                        severity_icon = "🔴"  # Severe - rosso
                    elif "non life-threatening immune-mediated" in last_part:
                        severity_icon = "🟠"  # Warning - arancione
                    elif "non life-threatening non immune-mediated" in last_part:
                        severity_icon = "🟡"  # Lenient warning - giallo
                    # Se c'è "none" o nessuna delle condizioni sopra, non aggiungiamo icona
                    
                    # Scrivi il testo con l'eventuale icona
                    if severity_icon:
                        result_container.write(f"{result_text}\n\n{severity_icon}")
                    else:
                        result_container.write(result_text)
                #st.subheader("Token usage")
                #st.json(tk)
            else:
                st.error(f"Error: {response.status_code}")


with tab2:
    st.header("Upload a CSV or Excel file")

    # File uploader
    uploaded_file = st.file_uploader("Choose the CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, dtype={"patient_id": str, "drug_code": str, "clinical_note": str})
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, dtype={"patient_id": str, "drug_code": str, "clinical_note": str})
            else:
                st.error("Error: Unsupported file type.")
                df = None
        except Exception as e:
            st.error(f"Error: Unable to read the file. Please, check the file format.")
            df = None
        
        if df is not None:
            filename = st.text_input("Enter filename (without extension) to save results", "processed_data")
            required_columns = {"patient_id", "drug_code", "clinical_note"}
            if not required_columns.issubset(df.columns):
                st.error(f"Error: The file must contain at least the following columns {required_columns}.")
            else:
                #df = df[list(required_columns)]  # Keep only required columns
                df["reaction_resp"] = ""  # Add a new column for reaction_resp
                df["response"] = ""  # Add a new column for response
                df["classification_resp"] = ""  # Add a new column for classification_resp
                df["timing"] = ""
                
                # Update NaN with empty strings
                df = df.fillna("")

                # Convert all columns to strings to avoid JSON compliance issues
                df = df.astype(str)
                
                # Create an interactive data viewer container
                df_container = st.empty()
                
                # Set column configurations
                column_config = {
                    'patient_id': st.column_config.TextColumn(label="ID"),
                    'drug_code': st.column_config.TextColumn(label="Drug Code"),
                    'clinical_note': st.column_config.TextColumn(label="Clinical note"),
                    'classification_resp': st.column_config.TextColumn(label="Classification"),
                    'reaction_resp': st.column_config.TextColumn(label="Reaction"),
                    'response': st.column_config.TextColumn(label="Response", width='large')  # Set the width to large
                }
                
                df_container.dataframe(df, column_config=column_config, use_container_width=True)  # Display the dataframe
                
                # Execute button
                if st.button("Execute"):
                    tkk = {'input':0, 'output':0, 'total':0}
                    for i, row in df.iterrows():
                        payload = {
                            "patient_id": row['patient_id'],
                            "drug_code": row['drug_code'],
                            "clinical_notes": row['clinical_note'],
                            "store": False  
                        }
                        
                        with st.spinner(f"Processing ID {row['patient_id']}..."):
                            try:
                                start_time = time.time()
                                response = requests.post(url, json=payload, stream=False)
                                end = time.time()
                            except requests.exceptions.RequestException as e:
                                st.error(f"Request Error: {e}")
                                continue
                            
                            err = True
                            if response.status_code == 200:
                                result_json = process_response(response)
                                if result_json:
                                    err = False
                                    result_row = row.to_dict()
                                    result_row['timing'] = end - start_time
                                    result_row['response'] = result_json.get('a', '')
                                    result_row['classification_resp'] = result_json.get('r', '')
                                    result_row['reaction_resp'] = result_json.get('rt', '')

                                    df.at[i, "response"] = result_row['response']
                                    df.at[i, "classification_resp"] = result_row['classification_resp']
                                    df.at[i, "reaction_resp"] = result_row['reaction_resp']
                                    df.at[i, "timing"] = str(result_row['timing']).replace(".",",")
                                    # Stylize the dataframe with updated data
                                    df_container.dataframe(df, column_config=column_config, use_container_width=True)  # Display the updated dataframe
                            
                            if err:
                                df.at[i, "response"] = f"Error: {response.status_code}"
                                # Stylize the dataframe with updated data
                                df_container.dataframe(df, column_config=column_config, use_container_width=True)  # Display the updated dataframe
 
                    
                    if filename:
                        if not filename.endswith('.xlsx'):
                            filename = f"{filename}.xlsx"
                        print(f"File name to save {filename}")

                        df.to_excel(filename, index=False)
                        st.success(f"File '{filename}' saved!", icon="✅")  
                   
                    #st.subheader("Token usage")
                    #st.json(tkk)