import streamlit as st
import pandas as pd
import pulp
import base64
import io
import random

st.title("DraftKings Showdown Lineup Optimizer")

# Create template download button
def create_template_csv():
    template_df = pd.DataFrame(columns=[
        'Name',             # Required
        'Team',             # Required
        'Position',         # Optional
        'Salary',          # Required
        'Projection',       # Required
        'Total Own',        # Required
        'CPT Salary',      # Required
        'CPT Projection',   # Required
        'CPT Own',         # Required
        'Ceiling',         # Required
        'Value'            # Optional
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
col1, col2 = st.columns([2,1])
with col1:
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
with col2:
    template_df = create_template_csv()
    st.markdown(
        download_link(template_df, 'template.csv', '📥 Download Template CSV'),
        unsafe_allow_html=True
    )

def optimize_lineup(df, previous_lineups=None, optimize_for='Projection', cpt_lock=None, use_random=False, random_weight=0.5, use_ownership_limit=False, max_ownership=None):
    if previous_lineups is None:
        previous_lineups = []
    
    # Create a copy of the dataframe
    df_random = df.copy()
    
    if use_random:
        for idx, row in df_random.iterrows():
            # Calculate randomized FLEX projection
            proj_range = float(row['Ceiling']) - float(row['Projection'])
            max_random = float(row['Projection']) + (proj_range * random_weight)
            df_random.at[idx, 'Projection'] = random.uniform(
                float(row['Projection']), 
                max_random
            )
            
            # Calculate randomized CPT projection
            cpt_proj = float(row['CPT Projection'])
            cpt_ceil = float(row['Ceiling']) * 1.5
            cpt_range = cpt_ceil - cpt_proj
            max_random_cpt = cpt_proj + (cpt_range * random_weight)
            df_random.at[idx, 'CPT Projection'] = random.uniform(
                cpt_proj,
                max_random_cpt
            )
    
    prob = pulp.LpProblem("DFS_Optimizer", pulp.LpMaximize)
    
    player_vars = {}
    for idx, row in df_random.iterrows():
        player_vars[f"{row['Name']}_FLEX"] = pulp.LpVariable(f"{row['Name']}_FLEX", 0, 1, pulp.LpBinary)
        player_vars[f"{row['Name']}_CPT"] = pulp.LpVariable(f"{row['Name']}_CPT", 0, 1, pulp.LpBinary)
    
    if optimize_for == 'Projection':
        prob += pulp.lpSum([
            player_vars[f"{row['Name']}_FLEX"] * float(row['Projection']) +
            player_vars[f"{row['Name']}_CPT"] * float(row['CPT Projection'])
            for idx, row in df_random.iterrows()
        ])
    else:  # Optimize for Ceiling
        prob += pulp.lpSum([
            player_vars[f"{row['Name']}_FLEX"] * float(row['Ceiling']) +
            player_vars[f"{row['Name']}_CPT"] * float(row['Ceiling']) * 1.5
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
    
    for prev_lineup in previous_lineups:
        prob += pulp.lpSum([
            player_vars[f"{player['Name']}_FLEX"] if player['Position'] == 'FLEX'
            else player_vars[f"{player['Name']}_CPT"]
            for player in prev_lineup
        ]) <= len(prev_lineup) - 1
    
    # Add ownership constraint if enabled
    if use_ownership_limit and max_ownership is not None:
        prob += pulp.lpSum([
            player_vars[f"{row['Name']}_FLEX"] * float(row['Total Own']) +
            player_vars[f"{row['Name']}_CPT"] * float(row['CPT Own'])
            for idx, row in df_random.iterrows()
        ]) <= max_ownership
    
    prob.solve()
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        return None, 0, 0, 0
    
    selected_players = []
    total_salary = 0
    total_points = 0
    total_ceiling = 0
    
    for idx, row in df.iterrows():  # Use original df for display
        flex_var = player_vars[f"{row['Name']}_FLEX"]
        cpt_var = player_vars[f"{row['Name']}_CPT"]
        
        if flex_var.value() == 1:
            selected_players.append({
                'Position': 'FLEX',
                'Name': row['Name'],
                'Team': row['Team'],
                'Pos': row['Position'],
                'Salary': float(row['Salary']),
                'Projected': float(row['Projection']),
                'Ceiling': float(row['Ceiling']),
                'Ownership': float(row['Total Own']),
                'Value': float(row['Value'])
            })
            total_salary += float(row['Salary'])
            total_points += float(row['Projection'])
            total_ceiling += float(row['Ceiling'])
            
        if cpt_var.value() == 1:
            selected_players.append({
                'Position': 'CPT',
                'Name': row['Name'],
                'Team': row['Team'],
                'Pos': row['Position'],
                'Salary': float(row['CPT Salary']),
                'Projected': float(row['CPT Projection']),
                'Ceiling': float(row['Ceiling']) * 1.5,
                'Ownership': float(row['CPT Own']),
                'Value': float(row['Value'])
            })
            total_salary += float(row['CPT Salary'])
            total_points += float(row['CPT Projection'])
            total_ceiling += float(row['Ceiling']) * 1.5
    
    results_df = pd.DataFrame(selected_players)
    results_df['Position_Order'] = results_df['Position'].map({'CPT': 0, 'FLEX': 1})
    results_df = results_df.sort_values(['Position_Order', 'Projected'], ascending=[True, False])
    results_df = results_df.drop('Position_Order', axis=1)
    
    return results_df, total_salary, total_points, total_ceiling

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
        
        # Convert numeric columns, only process columns that exist
        numeric_cols = ['Salary', 'Projection', 'Total Own', 'CPT Salary', 
                       'CPT Projection', 'CPT Own', 'Ceiling']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        st.write("Data Preview:")
        st.dataframe(df)
        
        # Add CPT lock dropdown
        cpt_lock = st.selectbox(
            'Lock in Captain (optional):',
            ['None'] + df['Name'].tolist()
        )
        
        # Add randomization controls
        with st.expander("Randomization Settings"):
            use_randomization = st.checkbox("Use Randomization", value=False)
            if use_randomization:
                randomization_weight = st.slider(
                    "Randomization Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="0 = Use Projections Only, 1 = Use Full Range to Ceiling"
                )
        
        # Add this in the settings area, after the randomization controls
        with st.expander("Ownership Settings"):
            use_ownership_limit = st.checkbox("Set Maximum Total Ownership", value=False)
            if use_ownership_limit:
                max_total_ownership = st.slider(
                    "Maximum Total Ownership %",
                    min_value=0,
                    max_value=600,  # 6 players * 100%
                    value=300,
                    step=5,
                    help="Set the maximum allowed total ownership percentage for the lineup"
                )
        
        if st.button("Generate Lineups"):
            # Generate projection-optimized lineups
            st.header("Top Lineups by Projection")
            projection_lineups = []
            
            for i in range(2):
                with st.spinner(f'Generating projection-optimized lineup {i+1}...'):
                    result = optimize_lineup(
                        df, 
                        projection_lineups, 
                        'Projection', 
                        cpt_lock,
                        use_randomization,
                        randomization_weight if use_randomization else 0.0,
                        use_ownership_limit,
                        max_total_ownership if use_ownership_limit else None
                    )
                    
                    if result[0] is None:
                        st.error(f"Could not find projection lineup #{i+1}")
                        break
                    
                    results_df, total_salary, total_points, total_ceiling = result
                    projection_lineups.append(results_df.to_dict('records'))
                    
                    st.subheader(f"Projection Lineup #{i+1}")
                    if cpt_lock != 'None':
                        st.write(f"Captain locked: {cpt_lock}")
                    
                    total_own = results_df['Ownership'].sum()
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Total Salary", f"${total_salary:,.2f}")
                    col2.metric("Projected Points", f"{total_points:.2f}")
                    col3.metric("Ceiling", f"{total_ceiling:.2f}")
                    col4.metric("Remaining Salary", f"${50000 - total_salary:,.2f}")
                    col5.metric("Total Ownership", f"{total_own:.1f}%")
                    
                    st.dataframe(results_df)
            
            # Generate ceiling-optimized lineups
            st.header("Top Lineups by Ceiling")
            ceiling_lineups = []
            
            for i in range(2):
                with st.spinner(f'Generating ceiling-optimized lineup {i+1}...'):
                    result = optimize_lineup(
                        df, 
                        ceiling_lineups, 
                        'Ceiling', 
                        cpt_lock,
                        use_randomization,
                        randomization_weight if use_randomization else 0.0,
                        use_ownership_limit,
                        max_total_ownership if use_ownership_limit else None
                    )
                    
                    if result[0] is None:
                        st.error(f"Could not find ceiling lineup #{i+1}")
                        break
                    
                    results_df, total_salary, total_points, total_ceiling = result
                    ceiling_lineups.append(results_df.to_dict('records'))
                    
                    st.subheader(f"Ceiling Lineup #{i+1}")
                    if cpt_lock != 'None':
                        st.write(f"Captain locked: {cpt_lock}")
                    
                    total_own = results_df['Ownership'].sum()
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Total Salary", f"${total_salary:,.2f}")
                    col2.metric("Projected Points", f"{total_points:.2f}")
                    col3.metric("Ceiling", f"{total_ceiling:.2f}")
                    col4.metric("Remaining Salary", f"${50000 - total_salary:,.2f}")
                    col5.metric("Total Ownership", f"{total_own:.1f}%")
                    
                    st.dataframe(results_df)
                    
    except Exception as e:
        st.error(f"Error: {str(e)}")