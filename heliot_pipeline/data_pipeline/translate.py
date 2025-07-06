import pandas as pd
from openai import OpenAI
import os

client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

def translate_to_italian(text):
    try:
        USER_ENGLISH_TRANSLATION ="""Translate in Italian from English: {text}
Report only the translation, nothing else. If you don't know the translation, report the original text. Do not change the drug names."""

        response = client.chat.completions.create(model="gpt-4o",
                                    messages=[{"role": "system", "content": ""},
                                            {"role": "user", "content":  USER_ENGLISH_TRANSLATION.format(text=text)}],
                                    max_tokens=3000,
                                    temperature = 0)
        cleaned_text = response.choices[0].message.content
        return cleaned_text
    except Exception as e:
        print(f"Failed to process text with GPT-4 _translate_in_english: {e}")
        return None

def translate_clinical_notes(input_file, output_file):
    # Read the Excel file
    df = pd.read_excel(input_file)
    
    # Translate the 'clinical_note' field
    if 'clinical_note' in df.columns:
        df['clinical_note'] = df['clinical_note'].apply(translate_to_italian)
    else:
        print("Column 'clinical_note' not found in the Excel file.")
        return
    
    # Write DataFrame to new Excel file
    df.to_excel(output_file, index=False)

# Input and output file names
input_filename = './patients_synthetic.xlsx'  # replace with your file
output_filename = 'patients_synthetic_it.xlsx'

# Translate and save the file
translate_clinical_notes(input_filename, output_filename)

print("Translation completed and saved to", output_filename)