import streamlit as st
import pandas as pd
import pulp
import base64
import io
import random
import numpy as np

# Set page configuration
st.set_page_config(page_title="DraftKings Lineup Optimizer", layout="wide")

# Function to create a template CSV for DraftKings
def create_template_csv():
    template_df = pd.DataFrame(columns=[
        'Name',             # Required
        'Position',         # Optional
        'Team',             # Required
        'Salary',           # Required
        'CPT Salary',       # Required
        'Projection',       # Required
        'CPT Projection',    # Required
        'Total Own',        # Required
        'CPT Own',          # Required
        'Ceiling',          # Required
        'Value'             # Optional
    ])
    # Add one example row
    template_df.loc[0] = [
        'Jalen Hurts', 'QB', 'PHI', 10000, 15000, 10.0, 15.0, 30.75, 10.0, 25.0, 2.05
    ]
    return template_df

def create_template_csv_soccer():
    template_df = pd.DataFrame(columns=[
        'Name', 'POS', 'DK Salary', 'Projection', 'DK Team'
    ])
    # Add one example row
    template_df.loc[0] = [
        'Cole Palmer', 'M', 10000, 18.2, 'CHE'
    ]
    return template_df

# Function to download CSV link
def download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a href="{href}" download="{filename}">{text}</a>'

# Function to optimize DraftKings lineup
def optimize_lineup(df, cpt_lock=None, max_ownership=225, use_random=False, random_weight=0.5):
    # Create a copy of the dataframe
    df_random = df.copy()

    if use_random:
        # Add more randomization to ensure different lineups
        for idx, row in df_random.iterrows():
            # Randomize projections with wider variance
            proj_base = float(row['Projection'])
            proj_ceiling = float(row['Ceiling'])
            proj_range = proj_ceiling - proj_base
            
            # Increase randomization range
            max_random = proj_base + (proj_range * random_weight * 1.5)
            min_random = max(0, proj_base - (proj_range * random_weight * 0.5))
            df_random.at[idx, 'Projection'] = random.uniform(min_random, max_random)

            # Randomize CPT projections
            cpt_proj_base = float(row['CPT Projection'])
            cpt_ceiling = float(row['Ceiling']) * 1.5
            cpt_range = cpt_ceiling - cpt_proj_base
            
            max_random_cpt = cpt_proj_base + (cpt_range * random_weight * 1.5)
            min_random_cpt = max(0, cpt_proj_base - (cpt_range * random_weight * 0.5))
            df_random.at[idx, 'CPT Projection'] = random.uniform(min_random_cpt, max_random_cpt)
    else:
        # If not using randomization, add small random noise to break ties
        df_random['Projection'] = df_random['Projection'] + np.random.uniform(-0.01, 0.01, len(df_random))
        df_random['CPT Projection'] = df_random['CPT Projection'] + np.random.uniform(-0.01, 0.01, len(df_random))

    # Check for required columns
    required_cols = ['Name', 'Position', 'Team', 'Salary', 'CPT Salary', 'Projection', 'CPT Projection',
                     'Total Own', 'CPT Own', 'Ceiling']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

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
    
    # Process selected players
    for idx, row in df.iterrows():  # Use original df for display
        flex_var = player_vars[f"{row['Name']}_FLEX"]
        cpt_var = player_vars[f"{row['Name']}_CPT"]
        
        if cpt_var.value() == 1:
            selected_players.insert(0, {  # Insert CPT at the top
                'Position': 'CPT',
                'Name': row['Name'],
                'Team': row['Team'],
                'Salary': float(row['CPT Salary']),
                'Projected': float(row['CPT Projection']),
            })
            total_salary += float(row['CPT Salary'])
            total_points += float(row['CPT Projection'])
        
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
    
    results_df = pd.DataFrame(selected_players)
    return results_df, total_salary, total_points

# Function to optimize soccer lineup
def optimize_soccer_lineup(df):
    """
    Optimize soccer lineup with exact column names from the input file
    """
    # Check for required columns
    required_columns = ['Name', 'POS', 'DK Salary', 'Projection', 'DK Team']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Input DataFrame is missing required columns.")
    
    # Convert salary to numeric, removing any currency symbols and commas
    df['DK Salary'] = pd.to_numeric(df['DK Salary'].replace(r'[\$,]', '', regex=True))
    df['Projection'] = pd.to_numeric(df['Projection'])
    
    # Create optimization problem
    prob = pulp.LpProblem("Soccer_Optimizer", pulp.LpMaximize)
    
    # Create binary variables for each player
    player_vars = {}
    for idx, row in df.iterrows():
        player_vars[row['Name']] = pulp.LpVariable(f"player_{row['Name']}", 0, 1, pulp.LpBinary)
    
    # Objective: Maximize projection
    prob += pulp.lpSum([player_vars[row['Name']] * row['Projection'] for _, row in df.iterrows()])
    
    # Salary cap constraint
    prob += pulp.lpSum([player_vars[row['Name']] * row['DK Salary'] for _, row in df.iterrows()]) <= 50000
    
    # Position constraints
    defenders = df[df['POS'] == 'D']['Name'].tolist()
    midfielders = df[df['POS'].str.contains('M') & ~df['POS'].str.contains('M (M/F)')]['Name'].tolist()  # Exclude "M (M/F)"
    forwards = df[df['POS'] == 'F']['Name'].tolist()
    mf_midfielders = df[df['POS'] == 'M (M/F)']['Name'].tolist()  # Midfielders that can also play forward
    goalkeepers = df[df['POS'] == 'GK']['Name'].tolist()  # Goalkeepers
    
    # Total must be exactly 7 players
    prob += pulp.lpSum(player_vars.values()) == 7
    
    # Minimum 2 Defenders
    prob += pulp.lpSum(player_vars[player] for player in defenders) >= 2
    
    # Maximum 3 Defenders
    prob += pulp.lpSum(player_vars[player] for player in defenders) <= 3
    
    # Minimum 2 Midfielders (M) + Midfielders (M (M/F))
    prob += pulp.lpSum(player_vars[player] for player in midfielders) + \
            pulp.lpSum(player_vars[player] for player in mf_midfielders) >= 2
    
    # Minimum 2 Forwards (F) + Midfielders (M (M/F))
    prob += pulp.lpSum(player_vars[player] for player in forwards) + \
            pulp.lpSum(player_vars[player] for player in mf_midfielders) >= 2
    
    # Maximum 5 Midfielders (M (M/F))
    prob += pulp.lpSum(player_vars[player] for player in mf_midfielders) <= 5
    
    # Ensure exactly 1 goalkeeper
    prob += pulp.lpSum(player_vars[player] for player in goalkeepers) == 1
    
    # Solve
    prob.solve()
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise Exception("Could not find an optimal solution")
    
    # Get results
    selected_players = []
    total_salary = 0
    total_points = 0
    
    # Process selected players
    for idx, row in df.iterrows():
        if player_vars[row['Name']].value() == 1:
            selected_players.append({
                'Position': row['POS'],
                'Name': row['Name'],
                'Team': row['DK Team'],
                'Salary': row['DK Salary'],
                'Projection': row['Projection']
            })
            total_salary += row['DK Salary']
            total_points += row['Projection']
    
    return pd.DataFrame(selected_players), total_salary, total_points

# Main application
st.title("DraftKings Lineup Optimizer")

# Create tabs for different optimizers
tabs = st.tabs(["Showdown", "Soccer"])

# DraftKings Optimizer Tab
with tabs[0]:
    st.header("Showdown")
    # Add template download and file upload section
    col1, col2 = st.columns([2, 1])
    with col1:
        template_df = create_template_csv() 
        st.markdown(
            download_link(template_df, 'template.csv', '📥 Download Showdown Template'),
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader("Upload your Showdown CSV", type="csv")           
    # DraftKings optimizer logic here...
    if uploaded_file is not None:
        try:
            # Load the DataFrame
            df = pd.read_csv(uploaded_file)

            # Check for required columns
            required_cols = ['Name', 'Position', 'Team', 'Salary', 'Projection', 
                             'CPT Salary', 'CPT Projection', 'Total Own', 'CPT Own', 'Ceiling']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

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
                seen_lineups = set()  # To track unique lineups
                attempts = 0
                max_attempts = 500  # Increase maximum attempts
                
                # Add progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                while len(all_lineups) < 20 and attempts < max_attempts:
                    status_text.text(f"Generated {len(all_lineups)} unique lineups... (Attempt {attempts + 1})")
                    progress_bar.progress(min(len(all_lineups) / 20, 1.0))
                    
                    # Force randomization on after certain attempts if we're not getting enough unique lineups
                    force_random = attempts > 50 and len(all_lineups) < 10
                    current_random = use_randomization or force_random
                    current_weight = randomization_weight if not force_random else min(randomization_weight + 0.2, 1.0)
                    
                    result = optimize_lineup(df, cpt_lock, max_ownership, current_random, current_weight)
                    if result[0] is not None:
                        results_df, total_salary, total_points = result
                        
                        # Create a tuple of player names to check for uniqueness
                        lineup_key = tuple(sorted(results_df['Name'].tolist()))
                        
                        if lineup_key not in seen_lineups:
                            seen_lineups.add(lineup_key)
                            all_lineups.append((results_df, total_points))
                    
                    attempts += 1
                
                progress_bar.progress(1.0)
                status_text.text(f"Generated {len(all_lineups)} unique lineups")
                
                if len(all_lineups) < 20:
                    st.warning(f"Could only generate {len(all_lineups)} unique lineups within constraints")
                
                # Sort lineups by total projection (descending)
                all_lineups.sort(key=lambda x: x[1], reverse=True)
                
                # Show top 3 lineups in columns
                st.header("Top 3 Lineups by Projection")
                cols = st.columns(3)  # Create 3 columns
                for i in range(min(3, len(all_lineups))):
                    with cols[i]:
                        st.subheader(f"Lineup #{i + 1} (Proj: {all_lineups[i][1]:.2f})")
                        st.dataframe(all_lineups[i][0])

                # Create CSV of top 20 lineups with dynamic naming
                lineup_data = []
                for i, (lineup, projection) in enumerate(all_lineups):
                    lineup_row = {
                        'Lineup': i + 1,
                        'CPT': lineup['Name'].iloc[0],
                        'FLEX 1': lineup['Name'].iloc[1],
                        'FLEX 2': lineup['Name'].iloc[2],
                        'FLEX 3': lineup['Name'].iloc[3],
                        'FLEX 4': lineup['Name'].iloc[4],
                        'FLEX 5': lineup['Name'].iloc[5],
                        'Total Salary': lineup['Salary'].sum(),
                        'Total Projection': projection,
                    }
                    lineup_data.append(lineup_row)

                lineup_df = pd.DataFrame(lineup_data)
                # Dynamic naming for the CSV file
                csv_filename = f"{cpt_lock}_20_lineups.csv" if cpt_lock != 'None' else "top_20_lineups.csv"
                csv_link = download_link(lineup_df, csv_filename, f"📥 Download {csv_filename}")
                st.markdown(csv_link, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Soccer Optimizer Tab
with tabs[1]:
    st.header("Soccer")
    
    # Add template download section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            download_link(create_template_csv_soccer(), 'soccer_template.csv', '📥 Download Soccer Template'),
            unsafe_allow_html=True
        )
    
    uploaded_soccer_file = st.file_uploader("Upload your Soccer CSV", type="csv")
    
    if uploaded_soccer_file is not None:
        try:
            df_soccer = pd.read_csv(uploaded_soccer_file)

            # Optimize soccer lineup
            results_df, total_salary, total_points = optimize_soccer_lineup(df_soccer)

            # Display results
            st.subheader("Optimized Soccer Lineup")
            st.dataframe(results_df)
            st.metric("Total Salary", f"${total_salary:,.2f}")
            st.metric("Total Projection", f"{total_points:.2f}")

            # Show top 3 lineups for soccer
            st.header("Top 3 Soccer Lineups")
            cols = st.columns(3)  # Create 3 columns
            for i in range(min(3, len(results_df))):
                with cols[i]:
                    st.subheader(f"Lineup #{i + 1}")
                    st.dataframe(results_df.iloc[[i]])  # Display each lineup

        except Exception as e:
            st.error(f"Error: {str(e)}")