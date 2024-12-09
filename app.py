import streamlit as st
import pandas as pd
import pulp
import base64
import io
import random

# Set page configuration
st.set_page_config(page_title="DraftKings Lineup Optimizer", layout="wide")

st.title("DraftKings Lineup Optimizer")

# Create template download button
def create_template_csv():
    template_df = pd.DataFrame(columns=[
        'Name',             # Required
        'Team',             # Required
        'Position',         # Optional
        'Salary',           # Required
        'Projection',       # Required
        'Total Own',        # Required
        'CPT Salary',       # Required
        'CPT Projection',    # Required
        'CPT Own',          # Required
        'Ceiling',          # Required
        'Value'             # Optional
    ])
    # Add one example row
    template_df.loc[0] = [
        'Player Name', 'PHI', 'QB', 10000, 20.5, 15.0,
        15000, 30.75, 10.0, 25.0, 2.05
    ]
    return template_df

def download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a href="{href}" download="{filename}">{text}</a>'

# Add template download and file upload section
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
with col2:
    template_df = create_template_csv()
    st.markdown(
        download_link(template_df, 'template.csv', '📥 Download Template CSV'),
        unsafe_allow_html=True
    )

def optimize_lineup(df, cpt_lock=None, max_ownership=225, use_random=False, random_weight=0.5):
    # Create a copy of the dataframe
    df_random = df.copy()

    if use_random:
        for idx, row in df_random.iterrows():
            # Randomize projections
            proj_base = float(row['Projection'])
            proj_ceiling = float(row['Ceiling'])
            proj_range = proj_ceiling - proj_base
            
            max_random = proj_base + (proj_range * random_weight)
            min_random = proj_base - (proj_range * random_weight)
            df_random.at[idx, 'Projection'] = random.uniform(min_random, max_random)

            cpt_proj_base = float(row['CPT Projection'])
            cpt_ceiling = float(row['Ceiling']) * 1.5
            cpt_range = cpt_ceiling - cpt_proj_base
            
            max_random_cpt = cpt_proj_base + (cpt_range * random_weight)
            min_random_cpt = cpt_proj_base - (cpt_range * random_weight)
            df_random.at[idx, 'CPT Projection'] = random.uniform(min_random_cpt, max_random_cpt)

    prob = pulp.LpProblem("Showdown_Optimizer", pulp.LpMaximize)
    
    player_vars = {}
    for idx, row in df_random.iterrows():
        player_vars[f"{row['Name']}_FLEX"] = pulp.LpVariable(f"{row['Name']}_FLEX", 0, 1, pulp.LpBinary)
        player_vars[f"{row['Name']}_CPT"] = pulp.LpVariable(f"{row['Name']}_CPT", 0, 1, pulp.LpBinary)

    prob += pulp.lpSum([
        player_vars[f"{row['Name']}_FLEX"] * float(row['Projection']) +
        player_vars[f"{row['Name']}_CPT"] * float(row['CPT Projection'])
        for idx, row in df_random.iterrows()
    ])

    prob += pulp.lpSum([
        player_vars[f"{row['Name']}_FLEX"] * float(row['Salary']) +
        player_vars[f"{row['Name']}_CPT"] * float(row['CPT Salary'])
        for idx, row in df_random.iterrows()
    ]) <= 50000

    prob += pulp.lpSum([player_vars[f"{row['Name']}_FLEX"] for idx, row in df_random.iterrows()]) == 5
    prob += pulp.lpSum([player_vars[f"{row['Name']}_CPT"] for idx, row in df_random.iterrows()]) == 1

    for idx, row in df_random.iterrows():
        prob += player_vars[f"{row['Name']}_FLEX"] + player_vars[f"{row['Name']}_CPT"] <= 1

    if cpt_lock and cpt_lock != 'None':
        prob += player_vars[f"{cpt_lock}_CPT"] == 1

    # Add ownership constraint
    prob += pulp.lpSum([
        player_vars[f"{row['Name']}_FLEX"] * float(row['Total Own']) +
        player_vars[f"{row['Name']}_CPT"] * float(row['CPT Own'])
        for idx, row in df_random.iterrows()
    ]) <= max_ownership

    prob.solve()
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        return None, 0, 0
    
    selected_players = []
    total_salary = 0
    total_points = 0
    
    for idx, row in df.iterrows():  # Use original df for display
        flex_var = player_vars[f"{row['Name']}_FLEX"]
        cpt_var = player_vars[f"{row['Name']}_CPT"]
        
        if flex_var.value() == 1:
            selected_players.append({
                'Position': 'FLEX',
                'Name': row['Name'],
                'Team': row['Team'],
                'Salary': float(row['Salary']),
                'Projected': float(row['Projection']),
            })
            total_salary += float(row['Salary'])
            total_points += float(row['Projection'])
            
        if cpt_var.value() == 1:
            selected_players.append({
                'Position': 'CPT',
                'Name': row['Name'],
                'Team': row['Team'],
                'Salary': float(row['CPT Salary']),
                'Projected': float(row['CPT Projection']),
            })
            total_salary += float(row['CPT Salary'])
            total_points += float(row['CPT Projection'])
    
    results_df = pd.DataFrame(selected_players)
    return results_df, total_salary, total_points

# Showdown Tab
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Check for required columns
        required_cols = ['Name', 'Team', 'Position', 'Salary', 'Projection', 
                         'Total Own', 'CPT Salary', 'CPT Projection', 'CPT Own', 'Ceiling']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        # Add any missing optional columns with default values
        if 'Value' not in df.columns:
            df['Value'] = 0.0
        
        # Convert numeric columns
        numeric_cols = ['Salary', 'Projection', 'Total Own', 'CPT Salary', 
                        'CPT Projection', 'CPT Own', 'Ceiling']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # CPT lock dropdown
        cpt_lock = st.selectbox('Lock in Captain (optional):', ['None'] + df['Name'].tolist())
        
        # Maximum ownership input
        max_ownership = st.slider("Set Maximum Ownership %", min_value=0, max_value=600, value=225, step=5)

        # Randomization controls
        use_randomization = st.checkbox("Use Randomization", value=False)
        randomization_weight = 0.5  # Default value
        if use_randomization:
            randomization_weight = st.slider("Randomization Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

        if st.button("Generate Lineups"):
            all_lineups = []
            for i in range(20):  # Generate top 20 lineups
                result = optimize_lineup(df, cpt_lock, max_ownership, use_randomization, randomization_weight)
                if result[0] is not None:
                    results_df, total_salary, total_points = result
                    all_lineups.append(results_df)

            # Show top 3 lineups in columns
            st.header("Top 3 Lineups")
            cols = st.columns(3)  # Create 3 columns
            for i in range(min(3, len(all_lineups))):
                with cols[i]:
                    st.subheader(f"Lineup #{i + 1}")
                    st.dataframe(all_lineups[i])

            # Create CSV of top 20 lineups
            lineup_data = []
            for i, lineup in enumerate(all_lineups):
                lineup_row = {
                    'Lineup': i + 1,
                    'CPT': lineup['Name'].iloc[0],
                    'FLEX 1': lineup['Name'].iloc[1],
                    'FLEX 2': lineup['Name'].iloc[2],
                    'FLEX 3': lineup['Name'].iloc[3],
                    'FLEX 4': lineup['Name'].iloc[4],
                    'FLEX 5': lineup['Name'].iloc[5],
                    'Total Salary': lineup['Salary'].sum(),
                    'Total Projection': lineup['Projected'].sum(),
                }
                lineup_data.append(lineup_row)

            lineup_df = pd.DataFrame(lineup_data)
            csv_link = download_link(lineup_df, "top_20_lineups.csv", "📥 Download Top 20 Lineups (CSV)")
            st.markdown(csv_link, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")