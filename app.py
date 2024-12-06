import streamlit as st
import pandas as pd
import pulp
import random
import base64
import io

# Set the page configuration to wide layout
st.set_page_config(page_title="DraftKings Lineup Optimizer", layout="wide")

def optimize_soccer_lineup(df):
    """
    Optimize soccer lineup with exact column names from the input file
    """
    # Check for required columns
    required_columns = ['Name', 'POS', 'DK Salary', 'Projection', 'DK Team']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Input DataFrame is missing required columns.")
    
    # Convert salary to numeric, removing any currency symbols and commas
    df['DK Salary'] = pd.to_numeric(df['DK Salary'].replace('[\$,]', '', regex=True))
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
    forwards = df[df['POS'].str.contains('F')]['Name'].tolist()
    midfielders = df[df['POS'].str.contains('M') & ~df['POS'].str.contains('M (M/F)')]['Name'].tolist()  # Exclude "M (M/F)"
    defenders = df[df['POS'] == 'D']['Name'].tolist()
    mf_midfielders = df[df['POS'] == 'M (M/F)']['Name'].tolist()  # New position
    goalkeepers = df[df['POS'] == 'GK']['Name'].tolist()  # Goalkeepers
    
    # Total must be exactly 7 players
    prob += pulp.lpSum(player_vars.values()) == 7
    
    # Total number of M, F, and M (M/F) must be between 4 and 5
    prob += pulp.lpSum(player_vars[player] for player in midfielders) + \
            pulp.lpSum(player_vars[player] for player in forwards) + \
            pulp.lpSum(player_vars[player] for player in mf_midfielders) >= 4  # At least 4
    prob += pulp.lpSum(player_vars[player] for player in midfielders) + \
            pulp.lpSum(player_vars[player] for player in forwards) + \
            pulp.lpSum(player_vars[player] for player in mf_midfielders) <= 5  # At most 5
    
    # Allow up to 3 defenders
    prob += pulp.lpSum(player_vars[player] for player in defenders) <= 3  # At most 3 defenders
    # Ensure at least 2 defenders
    prob += pulp.lpSum(player_vars[player] for player in defenders) >= 2  # At least 2 defenders
    # Ensure exactly 1 goalkeeper
    prob += pulp.lpSum(player_vars[player] for player in goalkeepers) == 1  # Exactly 1 GK
    
    # Ensure at least 2 forwards (including mf_midfielders)
    prob += pulp.lpSum(player_vars[player] for player in forwards) + \
            pulp.lpSum(player_vars[player] for player in mf_midfielders) >= 2  # At least 2 forwards
    
    # Ensure at least 2 midfielders (including mf_midfielders)
    prob += pulp.lpSum(player_vars[player] for player in midfielders) + \
            pulp.lpSum(player_vars[player] for player in mf_midfielders) >= 2  # At least 2 midfielders
    
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
    
    # Sort by position
    position_order = {'GK': 1, 'D': 2, 'M': 3, 'F': 4, 'M (M/F)': 5}  # Sort by position
    selected_players.sort(key=lambda x: position_order[x['Position']])
    
    return pd.DataFrame(selected_players), total_salary, total_points

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

def optimize_lineup(df, previous_lineups=None, optimize_for='Projection', cpt_lock=None, use_random=False, random_weight=0.5, use_ownership_limit=False, max_ownership=None):
    if previous_lineups is None:
        previous_lineups = []
    
    # Create a copy of the dataframe
    df_random = df.copy()
    
    if use_random:
        for idx, row in df_random.iterrows():
            # Calculate range for FLEX projection
            proj_base = float(row['Projection'])
            proj_ceiling = float(row['Ceiling'])
            proj_range = proj_ceiling - proj_base
            proj_floor = proj_base - proj_range  # Mirror the ceiling range below the base
            
            # Calculate randomized range based on weight
            max_random = proj_base + (proj_range * random_weight)
            min_random = proj_base - (proj_range * random_weight)
            
            # Set randomized FLEX projection
            df_random.at[idx, 'Projection'] = random.uniform(
                min_random,
                max_random
            )
            
            # Calculate range for CPT projection
            cpt_proj_base = float(row['CPT Projection'])
            cpt_ceiling = float(row['Ceiling']) * 1.5
            cpt_range = cpt_ceiling - cpt_proj_base
            cpt_floor = cpt_proj_base - cpt_range  # Mirror the ceiling range below the base
            
            # Calculate randomized CPT range based on weight
            max_random_cpt = cpt_proj_base + (cpt_range * random_weight)
            min_random_cpt = cpt_proj_base - (cpt_range * random_weight)
            
            # Set randomized CPT projection
            df_random.at[idx, 'CPT Projection'] = random.uniform(
                min_random_cpt,
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

def main():
    st.title("DraftKings Lineup Optimizer")  # Set the title

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Showdown", "Soccer"])  # Showdown first, then Soccer

    # CSV upload section below the tabs
    template_df = create_template_csv()
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv", key="csv_uploader")  # Unique key
    with col2:
        st.markdown(
            download_link(template_df, 'template.csv', '📥 Download Template CSV'),
            unsafe_allow_html=True
        )

    with tab1:  # Showdown tab
        if uploaded_file is not None:
            try:
                # Try different encodings
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with a different encoding
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding='latin1')
                
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
                            help="0% = Use Base Projections Only\n" +
                                 "50% = Random range from (Base - Half Range) to (Base + Half Range)\n" +
                                 "100% = Random range from (Base - Full Range) to (Base + Full Range)\n" +
                                 "Range = Distance from Base to Ceiling"
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
                
                if st.button("Generate Showdown Lineup"):
                    # Generate all 20 lineups first
                    all_lineups = []
                    
                    for i in range(20):  # Generate all 20 lineups
                        with st.spinner(f'Generating lineup {i+1}/20...'):
                            result = optimize_lineup(
                                df, 
                                all_lineups, 
                                'Projection', 
                                cpt_lock,
                                use_randomization,
                                randomization_weight if use_randomization else 0.0,
                                use_ownership_limit,
                                max_total_ownership if use_ownership_limit else None
                            )
                            
                            if result[0] is None:
                                st.error(f"Could not find lineup #{i+1}")
                                break
                            
                            results_df, total_salary, total_points, total_ceiling = result
                            all_lineups.append(results_df.to_dict('records'))
                    
                    # Create CSV with all lineups
                    if all_lineups:
                        # Create a list to store the formatted lineup data
                        formatted_lineups = []
                        
                        for lineup in all_lineups:
                            lineup_row = {}
                            
                            # Sort lineup to ensure CPT is first, then FLEX players
                            sorted_lineup = sorted(lineup, key=lambda x: x['Position'] != 'CPT')  # CPT first, then FLEX
                            
                            # Add CPT
                            lineup_row['CPT'] = sorted_lineup[0]['Name']
                            
                            # Add FLEX players
                            for i, player in enumerate(sorted_lineup[1:], 1):
                                lineup_row[f'FLEX {i}'] = player['Name']
                            
                            # Add totals
                            lineup_row['Total Salary'] = sum(p['Salary'] for p in lineup)
                            lineup_row['Total Projection'] = sum(p['Projected'] for p in lineup)
                            lineup_row['Total Ceiling'] = sum(p['Ceiling'] for p in lineup)
                            
                            formatted_lineups.append(lineup_row)
                        
                        # Create DataFrame with specific column order
                        columns = ['CPT', 'FLEX 1', 'FLEX 2', 'FLEX 3', 'FLEX 4', 'FLEX 5', 
                                  'Total Salary', 'Total Projection', 'Total Ceiling']
                        csv_df = pd.DataFrame(formatted_lineups, columns=columns)
                        
                        # Create filename based on locked captain
                        filename = "20Lineups.csv"
                        if cpt_lock and cpt_lock != 'None':
                            filename = f"{cpt_lock}_20Lineups.csv"
                        
                        # Create download button for CSV with explicit encoding
                        csv_buffer = io.StringIO()
                        csv_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                        csv_str = csv_buffer.getvalue()
                        b64 = base64.b64encode(csv_str.encode('utf-8')).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download All Lineups (CSV)</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                        # Display only top 3 lineups
                        st.header("Top 3 Lineups by Projection")
                        for i in range(min(3, len(all_lineups))):
                            lineup = all_lineups[i]
                            results_df = pd.DataFrame(lineup)
                            total_salary = results_df['Salary'].sum()
                            total_points = results_df['Projected'].sum()
                            total_ceiling = results_df['Ceiling'].sum()
                            total_own = results_df['Ownership'].sum()
                            
                            st.subheader(f"Projection Lineup #{i+1}")
                            if cpt_lock != 'None':
                                st.write(f"Captain locked: {cpt_lock}")
                            
                            col1, col2, col3, col4, col5 = st.columns(5)
                            col1.metric("Total Salary", f"${total_salary:,.2f}")
                            col2.metric("Projected Points", f"{total_points:.2f}")
                            col3.metric("Ceiling", f"{total_ceiling:.2f}")
                            col4.metric("Remaining Salary", f"${50000 - total_salary:,.2f}")
                            col5.metric("Total Ownership", f"{total_own:.1f}%")
                            
                            st.dataframe(results_df)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    with tab2:  # Soccer tab
        if uploaded_file is not None:
            try:
                # Try different encodings
                try:
                    df_soccer = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with a different encoding
                    uploaded_file.seek(0)  # Reset file pointer
                    df_soccer = pd.read_csv(uploaded_file, encoding='latin1')
                
                # Check for required columns
                required_cols = ['Name', 'POS', 'DK Salary', 'Projection', 'DK Team']
                missing_cols = [col for col in required_cols if col not in df_soccer.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.stop()
                
                # Process the soccer lineup
                results_df, total_salary, total_points = optimize_soccer_lineup(df_soccer)
                
                st.write("Soccer Lineup Results:")
                st.dataframe(results_df)
                st.metric("Total Salary", f"${total_salary:,.2f}")
                st.metric("Total Projection", f"{total_points:.2f}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()