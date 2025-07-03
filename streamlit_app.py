import streamlit as st
import json
import pandas as pd
import time
from datetime import datetime
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env")
client = OpenAI()

# Configure page
st.set_page_config(
    page_title="Pamokų planavimo sistema",
    page_icon="📚",
    layout="wide"
)

uzd = {}

input_data_state = False
teacher_input_state = False
lesson_task_state = False
lesson_plan_state = False


CLASS_PROFILE_PATH = "data/el_dienynas/class_profile.json"
HISTORY_CURRICULUM_PATH = "data/teminiai_planai/istorija.csv"
BUP_COMPETENCIES_PATH = "data/BUP/Istorijos_BUP_kompetenciju_ugdymas.csv"
BUP_STUDY_CONTENT_PATH = "data/BUP/Istorijos_BUP_mokymosi_turinys.csv"
BUP_ACHIEVEMENTS_BY_SUBJECT_PATH = "data/BUP/istorijos_BUP_pasiekimai_pagal_sritis.csv"
ACTIVITIES_PATH = "data/veiklos_strukturos/visos_struktruros.json"
LESSON_PLAN_STRUCTURE_PATH = "data/pamokos_plano_struktura/PP_struktura.csv"
SOCIAL_SKILLS_PATH = "data/igudziai/bendravimo_igudziai.csv"


col_bup_kompetencijos = ['Dalykas', 'Ugdoma kompetencija', 'Sritis', 'Aprasymas']
col_bup_mokymosi_turinys = ['Dalykas', 'Turinio tema', 'Sritis-Tema', 'Mokimosi turinys']
col_bup_pasiekimai = ['Dalykas', 'Pasiekimo sritis', 'Pasiekimų lygis', 'Pasiekimas', 'Aprasas']
col_teminis_planas = ['Dalykas', 'Turinio sritis', 'Skyrius', 'Tema', 'Pasiekimas', 'Ugdoma kompetencija']




activities_filter_promt = """Sukurk mokiniui skirtą pamokos planą, kuris:
           tiksliai laikosi pamokos plano struktura nurodyta 'pamokos plano struktura',
           remiasi mokymosi uždaviniu "Mokymosi uždavinys",
           remiasi "veiklos stuktūra" turiniu ir apima jos pavadinimą"""


lesson_task_promt="""Sukurk mokiniui skirtą mokymosi uždavinį, kuris:
            aprašo, ką turi pasiekti mokinys pamokos pabaigoje ir pagal kokius kriterijus yra vertinama sėkmė,
            remiasi bendrųjų programų 'pasiekimo sritimi' ir 'pasiekimų lygiu',
            remiasi teminiame plane nurodyta 'Tema' ir 'Ugdomais pasiekimais'.
            remiasi parametru "Uždavinio formavimas", kuris nurodo ar užduotis turi būti altiekama individualiai ar grupėje
            Rezultatas turi atitikti struktūrą 1.Sąlyga, 2. Atlikimas, 3. Kriterijus, be papildomų apibendrinumų ir paaiškinimų kaip jis buvo pasiektas"""

lesson_plan_promt = """Sukurk mokiniui skirtą pamokos planą, kuris:
            tiksliai laikosi pamokos plano struktura nurodyta 'pamokos plano struktura',
            remiasi mokymosi uždaviniu "Mokymosi uždavinys",
            remiasi "veiklos stuktūra" turiniu ir apima jos pavadinimą. Sugeneruotame pamokos plano dalyje "VEIKLA", turi būti perkelti duomenys iš "Veiklos struktūra" skyriaus "Struktūros eiga", be jokių modifikacijų.
            remiasi "ugdoma kompetencija".
            remiasi "vadovėlio medžiaga"
            Aprašydamas veiklą nurodyk, kas dirbs pamokoje, pagal parametrą „Pamokoje_dirbs“.
            """

if 'init_input_data' not in st.session_state:
    st.session_state.init_input_data = None

if 'teacher_input_data' not in st.session_state:
    st.session_state.teacher_input_data = None

if 'lesson_task' not in st.session_state:
    st.session_state.lesson_task = None

if 'lesson_plan_generated' not in st.session_state:
    st.session_state.lesson_plan_generated = False

if 'generation_status' not in st.session_state:
    st.session_state.generation_status = False

if 'bup_data1' not in st.session_state:
    st.session_state.bup_data1 = None

if 'bup_data2' not in st.session_state:
    st.session_state.bup_data2 = None

if 'bup_data3' not in st.session_state:
    st.session_state.bup_data3 = None

if 'curiculum_data' not in st.session_state:
    st.session_state.curiculum_data = None

if 'activities_data' not in st.session_state:
    st.session_state.activities_data = None

if 'lesson_plan_structure_data' not in st.session_state:
    st.session_state.lesson_plan_structure_data = None

if 'tema_data' not in st.session_state:
    st.session_state.tema_data = None

if 'pp_str' not in st.session_state:
    st.session_state.pp_str = ""

if 'skillz' not in st.session_state:
    st.session_state.skillz = ""

if 'vadovelio_medziaga' not in st.session_state:
    st.session_state.vadovelio_medziaga = ""

# if "selected_topic" not in st.session_state:
#     st.session_state.selected_topic = list(data.keys())[0]

def get_bup_competencies(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        df.columns = df.columns.str.strip()
        # Clean text data - remove extra quotes from string columns
        text_columns = ['Dokumentas','Skyrius','Dalykas', 'Ugdoma kompetencija', 'Sritis', 'Aprasymas']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.strip('"')

        # Drop columns that are mostly empty (unnamed columns with mostly NaN)
        # Keep only columns with meaningful names or significant data
        cols_to_keep = []
        for col in df.columns:
            if not col.startswith('_') and col != '':
                cols_to_keep.append(col)
            elif df[col].notna().sum() > 0:  # Keep if has some non-null values
                cols_to_keep.append(col)

        df = df[cols_to_keep]

        # Rename unnamed columns to more descriptive names if they contain data
        column_mapping = {}
        for i, col in enumerate(df.columns):
            if col.startswith('_') or col == '':
                if df[col].notna().sum() > 0:
                    column_mapping[col] = f'Additional_Info_{i}'

        if column_mapping:
            df = df.rename(columns=column_mapping)

        # Remove rows that are completely empty
        df = df.dropna(how='all')

        df = df[['Dokumentas', 'Skyrius', 'Dalykas', 'Ugdoma kompetencija', 'Sritis', 'Aprasymas']]

        # Clean up text data further - handle potential encoding issues
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].replace('nan', pd.NA)

        logger.info(f"Successfully loaded curriculum data from {file_path}")
        logger.info(f"Curriculum data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        return df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error when reading {file_path}: {e}")
        # Try with different encoding
        try:
            logger.info("Trying with 'latin-1' encoding...")
            return read_csv_to_dataframe(file_path, encoding='latin-1')
        except:
            logger.error("Failed with alternative encoding as well")
            raise
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        raise

def get_bup_study_content(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    try:
        # Read CSV with proper encoding
        df = pd.read_csv(file_path, encoding=encoding)

        # Clean column names (remove extra whitespace)
        df.columns = df.columns.str.strip()

        # Expected columns for this curriculum content file
        expected_columns = ['Dokumentas','Skyrius', 'Dalykas', 'Klasė', 'Tema', 'Nr', 'Sritis-tema', 'Mokymosi turinys']

        # Verify all expected columns are present
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")

        # Clean text columns - remove extra quotes and whitespace
        text_columns = ['Dokumentas','Skyrius', 'Dalykas', 'Tema', 'Nr', 'Sritis-tema', 'Mokymosi turinys']
        for col in text_columns:
            if col in df.columns:
                # Remove outer quotes and extra whitespace
                df[col] = df[col].astype(str).str.strip().str.strip("'\"").str.strip()

        # Convert Klase (Class) to integer, handling any potential issues
        if 'Klasė' in df.columns:
            df['Klasė'] = pd.to_numeric(df['Klasė'], errors='coerce').astype('Int64')

        # Clean and standardize the Nr (Number) column
        if 'Nr' in df.columns:
            # Remove trailing dots and spaces, standardize format
            df['Nr'] = df['Nr'].str.rstrip('. ')

        # Remove any completely empty rows
        df = df.dropna(how='all')

        # Handle any potential encoding artifacts in text
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'Klasė':  # Skip the numeric column we converted
                df[col] = df[col].replace('nan', pd.NA)

        # Sort by class and topic number for logical ordering
        if 'Klasė' in df.columns and 'Nr' in df.columns:
            df = df.sort_values(['Klasė', 'Nr'], na_position='last')
            df = df.reset_index(drop=True)

        logger.info(f"Successfully loaded curriculum content from {file_path}")
        logger.info(f"Curriculum content shape: {df.shape}")
        logger.info(f"Classes covered: {sorted(df['Klasė'].dropna().unique()) if 'Klasė' in df.columns else 'N/A'}")
        logger.info(f"Subjects: {df['Dalykas'].unique().tolist() if 'Dalykas' in df.columns else 'N/A'}")

        return df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error when reading {file_path}: {e}")
        # Try with different encoding
        try:
            logger.info("Trying with 'latin-1' encoding...")
            return read_curriculum_content_csv(file_path, encoding='latin-1')
        except:
            logger.error("Failed with alternative encoding as well")
            raise
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        raise

def get_bup_achievements(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    try:
        # Read CSV with proper encoding
        df = pd.read_csv(file_path, encoding=encoding)

        # Clean column names (remove extra whitespace)
        df.columns = df.columns.str.strip()

        # Expected columns for this curriculum content file
        expected_columns = ['Dokumentas','Skyrius', 'Dalykas', 'Klasė', 'Tema', 'Nr', 'Sritis-tema', 'Mokymosi turinys']

        # Verify all expected columns are present
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")

        # Clean text columns - remove extra quotes and whitespace
        text_columns = ['Dokumentas','Skyrius', 'Dalykas', 'Tema', 'Nr', 'Sritis-tema', 'Mokymosi turinys']
        for col in text_columns:
            if col in df.columns:
                # Remove outer quotes and extra whitespace
                df[col] = df[col].astype(str).str.strip().str.strip("'\"").str.strip()

        # Convert Klase (Class) to integer, handling any potential issues
        if 'Klasė' in df.columns:
            df['Klasė'] = pd.to_numeric(df['Klasė'], errors='coerce').astype('Int64')

        # Clean and standardize the Nr (Number) column
        if 'Nr' in df.columns:
            # Remove trailing dots and spaces, standardize format
            df['Nr'] = df['Nr'].str.rstrip('. ')

        # Remove any completely empty rows
        df = df.dropna(how='all')

        # Handle any potential encoding artifacts in text
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'Klasė':  # Skip the numeric column we converted
                df[col] = df[col].replace('nan', pd.NA)

        # Sort by class and topic number for logical ordering
        if 'Klasė' in df.columns and 'Nr' in df.columns:
            df = df.sort_values(['Klasė', 'Nr'], na_position='last')
            df = df.reset_index(drop=True)

        logger.info(f"Successfully loaded curriculum content from {file_path}")
        logger.info(f"Curriculum content shape: {df.shape}")
        logger.info(f"Classes covered: {sorted(df['Klasė'].dropna().unique()) if 'Klasė' in df.columns else 'N/A'}")
        logger.info(f"Subjects: {df['Dalykas'].unique().tolist() if 'Dalykas' in df.columns else 'N/A'}")

        return df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error when reading {file_path}: {e}")
        # Try with different encoding
        try:
            logger.info("Trying with 'latin-1' encoding...")
            return read_curriculum_content_csv(file_path, encoding='latin-1')
        except:
            logger.error("Failed with alternative encoding as well")
            raise
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        raise

def get_curriculum(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    try:
        # Read CSV with proper encoding
        df = pd.read_csv(file_path, encoding=encoding)

        # Clean column names (remove extra whitespace)
        df.columns = df.columns.str.strip()

        # Convert numeric columns to appropriate types
        numeric_columns = [
            'Ugdymo proceso savaitė',
            'Skiriamų valandų skaičius 70 proc',
            'Skiriamų valandų skaičius 30 proc'
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"Successfully loaded history curriculum from {file_path}")
        logger.info(f"History curriculum shape: {df.shape}")

        return df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {file_path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading {file_path}: {e}")
        raise

def get_activities(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Read JSON file containing educational activities data to pandas DataFrame.

    Args:
        file_path (str): Path to the JSON file
        encoding (str): File encoding, defaults to 'utf-8'

    Returns:
        pd.DataFrame: Processed DataFrame with educational activities data
    """
    try:
        # Read JSON file
        with open(file_path, 'r', encoding=encoding) as file:
            data = json.load(file)

        # Extract activities list from the JSON structure
        activities = data.get('activities', [])

        if not activities:
            logger.warning("No activities found in the JSON file")
            return pd.DataFrame()

        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(activities)

        # Convert list columns to string format for better readability
        list_columns = ['strukturos_eiga', 'mokytojo_vaidmuo', 'specialiuju_poreikiu_mokiniai', 'pavyzdziai_temos']

        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ' | '.join(x) if isinstance(x, list) else x)

        logger.info(f"Successfully loaded activities data from {file_path}")
        logger.info(f"Activities data shape: {df.shape}")

        return df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error when reading {file_path}: {e}")
        # Try with different encoding
        try:
            logger.info("Trying with 'latin-1' encoding...")
            return read_json_to_dataframe(file_path, encoding='latin-1')
        except:
            logger.error("Failed with alternative encoding as well")
            raise
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        raise

def get_social_skills(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    try:
        # Read CSV with no headers and clean up quotes
        df = pd.read_csv(file_path, header=None, names=["Raw"], encoding=encoding)
        df["Raw"] = df["Raw"].str.strip('"').str.strip()

        # Extract code and description
        df[['Code', 'Bendravimo įgūdis']] = df["Raw"].str.extract(r'(\d+\.\d+)\s+(.*)')

        # Extract main theme (e.g., "Konstruktyviai komunikuoti tarpusavyje")
        df['Main Theme'] = df['Bendravimo įgūdis'].str.extract(r'^(.*?)(?:\s*-\s*)')

        # Reorder columns
        df = df[['Code', 'Main Theme', 'Bendravimo įgūdis']]

        logger.info(f"Successfully loaded competencies from {file_path}")
        logger.info(f"Competencies shape: {df.shape}")

        return df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {file_path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading {file_path}: {e}")
        raise

def get_lesson_plan_structure(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    try:
        # Read CSV with proper encoding
        df = pd.read_csv(file_path, encoding=encoding)

        # Clean column names (remove extra whitespace)
        df.columns = df.columns.str.strip()

        # Clean text columns - remove extra quotes and whitespace
        text_columns = ['Section', 'Description']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.strip('"').str.strip()

        # Convert Subsection to float, handling any potential issues
        if 'Subsection' in df.columns:
            df['Subsection'] = pd.to_numeric(df['Subsection'], errors='coerce')

        # Remove any completely empty rows
        df = df.dropna(how='all')

        # Sort by Section and Subsection for logical ordering
        if 'Section' in df.columns and 'Subsection' in df.columns:
            df = df.sort_values(['Section', 'Subsection'], na_position='last')
            df = df.reset_index(drop=True)

        logger.info(f"Successfully loaded PP structure from {file_path}")
        logger.info(f"PP structure shape: {df.shape}")

        return df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error when reading {file_path}: {e}")
        # Try with different encoding
        try:
            logger.info("Trying with 'latin-1' encoding...")
            return read_pp_struktura_csv(file_path, encoding='latin-1')
        except:
            logger.error("Failed with alternative encoding as well")
            raise
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        raise

def filter_data(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Filters a pandas DataFrame based on a dictionary of column-value filters.

    Parameters:
    - df (pd.DataFrame): The DataFrame to filter.
    - filters (dict): A dictionary where keys are column names and values are the values to filter by.

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """

    for column, value in filters.items():
        if column in df.columns:
            df = df[df[column] == value]
        # If the column does not exist, it is ignored
    return df

def filter_df_columns(df: pd.DataFrame, col: list) -> pd.DataFrame:
     return df[col]

def generate_lesson_task(df: pd.DataFrame, promt: str) -> str:
    response = client.responses.create(
        model="gpt-4.1",
        input=f"""{promt}. Argumentai: {df} """
    )

    print(response.output_text)
    return response.output_text

def generate_lesson_plan(df: pd.DataFrame, promt: str) -> str:
    response = client.responses.create(
        model="gpt-4.1",
        input=f"""{promt}. Argumentai: {df} """
    )
    return response.output_text

def df_to_json(name: str, df: pd.DataFrame, columns: list) -> dict:
    """
    Convert a pandas DataFrame to a JSON object with specified columns.

    Args:
        name (str): The key name for the JSON object
        df (pd.DataFrame): The pandas DataFrame to process
        columns (list): List of column names to filter

    Returns:
        dict: JSON object with the specified structure
    """
    # Filter the DataFrame to only include specified columns
    filtered_df = df[columns]

    # Convert each row to a dictionary with one key-value pair per dictionary
    result_list = []
    for _, row in filtered_df.iterrows():
        for col in columns:
            result_list.append({col: row[col]})

    # Return the final JSON structure
    return {name: result_list}



# Sidebar navigation
st.sidebar.title("📚 Pamokų planavimo sistema")
page = st.sidebar.radio(
    "Navigavimas:",
    ["Duomenų įvestis", "Pamokos planas", "Duomenys"], index=0
)

try:
    bup_data1_full = get_bup_competencies(BUP_COMPETENCIES_PATH)
    bup_data2_full = get_bup_study_content(BUP_STUDY_CONTENT_PATH)
    bup_data3_full = get_bup_achievements(BUP_ACHIEVEMENTS_BY_SUBJECT_PATH)
    curiculum_data_full = get_curriculum(HISTORY_CURRICULUM_PATH)
    activities_data_full = get_activities(ACTIVITIES_PATH)
    lesson_plan_structure_data_full = get_lesson_plan_structure(LESSON_PLAN_STRUCTURE_PATH)
    skills = get_social_skills(SOCIAL_SKILLS_PATH)
    st.session_state.init_input_data = True

except Exception as e:
    print(f"An unexpected error occurred loading data: {e}")

# Main content based on selected page
if page == "Duomenų įvestis":
    st.title("📋 Duomenų įvestis")
    st.header("Klasės profilis")

    with st.form("lesson_data_form"):
        col1, col2 = st.columns(2)

        with col1:
            mokslo_metai = st.selectbox(
                "Mokslo metai:",
                ["2024-2025", "2025-2026"], index=0
            )

            dalykas = st.selectbox(
                "Dalykas:",
                ["Matematika", "Lietuvių kalba ir literatūra", "Istorija"], index=2  # Default to "Istorija"
            )

            pamokoje_dirbs = st.selectbox(
                "Pamokoje dirbs:",
                ["Mokytojas", "Mokytojas ir spec. pedagogas", "Mokytojas ir mokinio padėjėjas"], index=0
            )

        with col2:
            klase = st.selectbox(
                "Klasė:",
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=4  # Default to 5
            )

            klase_pasirengimas = st.selectbox(
                "Klasės pasirengimas:",
                ["Aukštas pasirengimo lygis", "Vidutinis pasirengimo lygis", "Žemas pasirengimo lygis"], index=1
            )

            uzdavinio_formavimas = st.selectbox(
                "Uždavinio atlikimas:",
                ["Grupėje", "Individualiai"], index=0
            )

        igudis = st.selectbox(
            "Bendravimo įgūdis:",
            skills['Bendravimo įgūdis'], index=0
        )

        tema_options = curiculum_data_full['Tema'].dropna().unique()
        tema = st.selectbox("Tema:", options=tema_options)
        kompetencija = curiculum_data_full['Tema'].dropna().unique()


        st.session_state.vadovelio_medziaga = st.text_area(
            "Vadovėlio medžiaga", value="", height=100
        )

        filters_dict = {
            'Dalykas': dalykas,
            'Klasė': klase,
            'Mokslo metai': mokslo_metai,
            'Tema': tema,
            'Klasės pasirengimo lygis': klase_pasirengimas,
            'Bendravimo įgūdis': igudis,
            'Uždavinio formavimas': uzdavinio_formavimas,
            'Pamokoje dirbs': pamokoje_dirbs,
        }

        cur_data = filter_data(get_curriculum(HISTORY_CURRICULUM_PATH), filters_dict)
        if filters_dict['Dalykas'] == 'Istorija':
            filters_dict['Ugdoma kompetencija'] = cur_data['Ugdoma kompetencija'].values[0]
            filters_dict['Pasiekimas'] = cur_data['Pasiekimas'].values[0]

        logger.info(f"Updated filters:  {filters_dict}")
        bup_data3 = filter_data(get_bup_achievements(BUP_ACHIEVEMENTS_BY_SUBJECT_PATH), filters_dict)
        bup_data1 = filter_data(get_bup_competencies(BUP_COMPETENCIES_PATH), filters_dict)
        bup_data2 = filter_data(get_bup_study_content(BUP_STUDY_CONTENT_PATH), filters_dict)
        activities_data = filter_data(get_activities(ACTIVITIES_PATH), filters_dict)
        lesson_plan_structure_data = filter_data(get_lesson_plan_structure(LESSON_PLAN_STRUCTURE_PATH), filters_dict)
        skillz  = filter_data(get_social_skills(SOCIAL_SKILLS_PATH), filters_dict)

        submitted = st.form_submit_button("📝 Išsaugoti", use_container_width=True)

        if submitted:
            with st.spinner("Gaunami BUP duomenys ir teminiai planai..."):
                st.session_state.bup_data1 = bup_data1
                st.session_state.bup_data2 = bup_data2
                st.session_state.bup_data3 = bup_data3
                st.session_state.curiculum_data = filter_df_columns(cur_data, ["Dalykas", "Turinio sritis", "Tema"] )
                st.session_state.activities_data = activities_data
                st.session_state.lesson_plan_structure_data = lesson_plan_structure_data
                st.session_state.skillz = skillz
                st.session_state.teacher_input_data = True

                st.success("✅ BUP duomenys ir teminiai planai sėkmingai gauti!")
                st.rerun()

    if st.session_state.bup_data1 is not None:
        st.header("📊 Užkrauti duomenys")

        bup_data1 = get_bup_competencies(BUP_COMPETENCIES_PATH)
        bup_data2 = get_bup_study_content(BUP_STUDY_CONTENT_PATH)
        bup_data3 = get_bup_achievements(BUP_ACHIEVEMENTS_BY_SUBJECT_PATH)
        curiculum_data = get_curriculum(HISTORY_CURRICULUM_PATH)
        activities_data = get_activities(ACTIVITIES_PATH)
        lesson_plan_structure_data = get_lesson_plan_structure(LESSON_PLAN_STRUCTURE_PATH)
        skills = get_social_skills(SOCIAL_SKILLS_PATH)
        teacher_partnerships = pamokoje_dirbs
        teacher_input_state = True


        # st.json(filters_dict)
        # conv = df_to_json("Teminis planas", st.session_state.curiculum_data, ["Dalykas", "Turinio sritis", "Tema"])
        # st.subheader("Teminis planas dict")
        # st.json(df_to_json("BUP - Ugdomos kompetencijos", st.session_state.bup_data1, ["Dalykas", "Ugdoma kompetencija", "Sritis", "Aprasymas"] ))


        with st.expander("🔍 Peržiūrėti Teminio plano duomenis", expanded=False):
                st.subheader("Teminis planas")
                st.dataframe(st.session_state.curiculum_data, use_container_width=True)

        with st.expander("🔍 Peržiūrėti BUP duomenis", expanded=False):
            st.subheader("BUP - Ugdomos kompetencijos")
            st.dataframe(st.session_state.bup_data1, use_container_width=True)
            st.subheader("BUP - Mokymosi turinys")
            st.dataframe(st.session_state.bup_data2, use_container_width=True)
            st.subheader("BUP - Pasiekimai pagal sritis")
            st.dataframe(st.session_state.bup_data3, use_container_width=True)

        with st.expander("🔍 Peržiūrėti Veiklu rinkinio duomenis", expanded=False):
                st.subheader("Veiklu rinkinys")
                st.dataframe(st.session_state.activities_data, use_container_width=True)

        with st.expander("🔍 Peržiūrėti Pamokos plano strukturos duomenis", expanded=False):
                st.subheader("Pamokos plano struktura")
                st.dataframe(st.session_state.lesson_plan_structure_data, use_container_width=True)

        with st.expander("🔍 Peržiūrėti Bendravimo igudzius", expanded=False):
            st.subheader("Bendravimo igudziai")
            st.dataframe(st.session_state.skillz, use_container_width=True)

        with st.expander("🔍 Vadovėlio medžiaga", expanded=False):
            if st.session_state.vadovelio_medziaga != "":
                st.markdown(st.session_state.vadovelio_medziaga)
            else:
              st.markdown("Nėra")

        if st.button("🚀 Generuoti uždavinį", use_container_width=True) :
            with st.spinner("Generuojamas uždavinys..."):
                st.session_state.bup_df = filter_data(get_bup_achievements(BUP_ACHIEVEMENTS_BY_SUBJECT_PATH), filters_dict)
                st.session_state.cur_df = filter_data(get_curriculum(HISTORY_CURRICULUM_PATH), filters_dict)

                args_df = {
                    "Bendroji ugdymo programa": st.session_state.bup_df ,
                    "Teminis planas": st.session_state.cur_df,
                    'Uždavinio formavimas': uzdavinio_formavimas,
                }

                st.session_state.lesson_task = generate_lesson_task(args_df, lesson_task_promt)
                lesson_task_state = True

        if st.session_state.lesson_task:

                with st.expander("🔍 Input data", expanded=False):
                    st.subheader("Promt")
                    st.markdown(lesson_task_promt, unsafe_allow_html=True)
                    st.subheader("BUP - Pasiekimai pagal sritis")
                    st.dataframe(st.session_state.bup_df, use_container_width=True)
                    st.subheader("Teminis planas")
                    st.dataframe(st.session_state.cur_df, use_container_width=True)
                    st.markdown(f"**Uždavinio formavimas**: {uzdavinio_formavimas}", unsafe_allow_html=True)

                with st.expander("🔍 Peržiūrėti mokymosi uždavinį", expanded=True):
                    st.subheader("Sugeneruotas Mokymosi uždavinys:")
                    st.markdown(st.session_state.lesson_task)


        if st.button("🚀 Generuoti pamokos planą", use_container_width=True) and st.session_state.lesson_task:
            with st.spinner("Generuojamas pamokos planas..."):
                bup_df = filter_data(get_bup_achievements(BUP_ACHIEVEMENTS_BY_SUBJECT_PATH), filters_dict)
                cur_df = filter_data(get_curriculum(HISTORY_CURRICULUM_PATH), filters_dict)
                st.session_state.activities_data = filter_data(get_activities(ACTIVITIES_PATH), filters_dict)


                args_df = {
                    "Pamokos plano struktura": lesson_plan_structure_data,
                    "Teminis planas": cur_df,
                    "Mokymosi uzdavinys": st.session_state.lesson_task,
                    "Veiklos struktura": st.session_state.activities_data,
                    "Pamokoje dirbs": pamokoje_dirbs,
                    "Vadovėlio medžiaga": st.session_state.vadovelio_medziaga,

                }

                st.session_state.pp_str =  generate_lesson_plan(args_df, lesson_plan_promt)
                lesson_plan_state = True

                with st.expander("🔍 Peržiūrėti pamokos plana", expanded=False):
                    # st.subheader("PP input")
                    # st.json(args_df)
                    # st.subheader("PP")
                    st.markdown(st.session_state.pp_str)

                st.session_state.lesson_plan_generated = True
                st.session_state.generation_status = f"Pamokos planas sėkmingai sugeneruotas!"

        # Show generation status
        if st.session_state.generation_status:
            if st.session_state.lesson_plan_generated:
                st.success(f"✅ {st.session_state.generation_status}")
                st.info("👆 Eikite į 'Pamokos planas' skyrių, kad peržiūrėtumėte sugeneruotą planą")
            else:
                st.error(f"❌ {st.session_state.generation_status}")


elif page == "Pamokos planas":
    if st.session_state.pp_str:
        with st.expander("🔍 PP įeities duomenys ", expanded=False):
            st.subheader("Promt")
            st.markdown(lesson_plan_promt, unsafe_allow_html=True)
            st.subheader("Veiklos struktura")
            st.dataframe(st.session_state.activities_data, use_container_width=True)

            st.subheader("Mokymosi uždavinys")
            st.markdown(st.session_state.lesson_task)

            st.subheader("Vadovėlio medžiaga")
            st.markdown(st.session_state.vadovelio_medziaga)


        st.title("📖 **Pamokos Planas**")
        st.markdown(st.session_state.pp_str)
        st.markdown("---")
    else:
        st.info("ℹ️ Pamokos planas dar nesugeneruotas")


elif page == "Duomenys":
    st.title("📖 Užkrauti duomenys")

    with st.expander("🔍 Peržiūrėti BUP duomenis", expanded=False):
        st.subheader("BUP - Ugdomos kompetencijos")
        st.dataframe(bup_data1_full, use_container_width=True)

        st.subheader("BUP - Mokymosi turinys")
        st.dataframe(bup_data2_full, use_container_width=True)

        st.subheader("BUP - Pasiekimai pagal sritis")
        st.dataframe(bup_data3_full, use_container_width=True)

    with st.expander("🔍 Peržiūrėti Teminio plano duomenis", expanded=False):
        st.subheader("Teminis planas")
        st.dataframe(curiculum_data_full, use_container_width=True)

    with st.expander("🔍 Peržiūrėti Veiklu rinkinio duomenis", expanded=False):
        st.subheader("Veiklu rinkinys")
        st.dataframe(activities_data_full, use_container_width=True)

    with st.expander("🔍 Peržiūrėti Pamokos plano strukturos duomenis", expanded=False):
        st.subheader("Pamokos plano struktura")
        st.dataframe(lesson_plan_structure_data_full, use_container_width=True)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Statusai**")

input_data_state_icon = ":white_check_mark:" if st.session_state.init_input_data else ":exclamation:"
teacher_input_state_icon = ":white_check_mark:" if st.session_state.teacher_input_data else ":exclamation:"
lesson_task_state_icon = ":white_check_mark:" if lesson_task_state else ":exclamation:"
lesson_plan_state_icon = ":white_check_mark:" if lesson_plan_state else ":exclamation:"

st.sidebar.write(f"Įeities duomenys  {input_data_state_icon}")
st.sidebar.write(f"Mokytojo įvedami duomenys  {teacher_input_state_icon}")
st.sidebar.write(f"Mokymosi uždavinys  {lesson_task_state_icon}")
st.sidebar.write(f"Pamokos planas  {lesson_plan_state_icon}")

st.sidebar.markdown("---")
st.sidebar.markdown("Versija 1.3")
st.sidebar.markdown("Atnaujinta 2025.03.03, 10:05")
