import streamlit as st
import pandas as pd
import pulp
import random
import altair as alt  # Import Altair for more customizable charts

# Set the page configuration to wide layout
st.set_page_config(page_title="Draftkings Lineup Optimizer", layout="wide")

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

def main():
    st.title("Sports Lineup Optimizer")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Showdown", "Soccer"])
    
    with tab1:
        st.header("Showdown Tab")
        
        # Define the uploaded file for Showdown
        uploaded_file_showdown = st.file_uploader("Upload Showdown player pool", type=["csv", "xlsx", "xls"], key="showdown_upload")
        
        if uploaded_file_showdown is not None:
            # Your existing Showdown tab code goes here
            pass  # Replace with your Showdown logic
        
        # Download Template CSV link
        st.markdown("[Download Template CSV](#)")  # Replace # with the actual link to your template

    with tab2:
        st.header("⚽ Soccer Lineup Optimizer")  # Added soccer ball emoji
        
        # Download Template CSV link for Soccer
        st.markdown("[Download Template CSV](#)")  # Replace # with the actual link to your template
        
        # Define the uploaded file for Soccer
        uploaded_file_soccer = st.file_uploader("Upload soccer player pool", type=["csv", "xlsx", "xls"], key="soccer_upload")
        
        if uploaded_file_soccer is not None:
            try:
                # Read the file based on its type
                file_extension = uploaded_file_soccer.name.split('.')[-1].lower()
                if file_extension == 'csv':
                    df_soccer = pd.read_csv(uploaded_file_soccer)
                else:  # xlsx or xls
                    df_soccer = pd.read_excel(uploaded_file_soccer)
                
                # Clean column names - remove spaces and convert to proper format
                df_soccer.columns = df_soccer.columns.str.strip()
                
                # Show the uploaded data
                st.subheader("Uploaded Player Pool")
                st.dataframe(df_soccer)
                
                # Optimize button
                if st.button("Generate Top 3 Soccer Lineups"):
                    with st.spinner('Optimizing soccer lineups...'):
                        lineups = []
                        for i in range(10):  # Generate multiple lineups to find the top 3
                            random.seed(i)  # Set a different seed for each lineup
                            
                            # Create a modified DataFrame for randomization
                            modified_df = df_soccer.copy()
                            
                            # Randomly adjust player projections
                            for index, row in modified_df.iterrows():
                                # Randomly adjust the projection by a small amount
                                adjustment = random.uniform(-1, 1)  # Adjust by -1 to +1
                                modified_df.at[index, 'Projection'] += adjustment
                            
                            # Ensure projections remain non-negative
                            modified_df['Projection'] = modified_df['Projection'].clip(lower=0)
                            
                            # Generate the optimal lineup
                            optimal_lineup, total_salary, total_points = optimize_soccer_lineup(modified_df)
                            lineups.append((optimal_lineup, total_salary, total_points))
                        
                        # Sort lineups by total projected points
                        lineups.sort(key=lambda x: x[2], reverse=True)  # Sort by projected points (index 2)
                        
                        # Display the top 3 lineups
                        cols = st.columns(3)  # Create 3 columns
                        
                        for i in range(3):  # Display only the top 3 lineups
                            lineup, salary, points = lineups[i]
                            with cols[i]:  # Use the ith column
                                st.subheader(f"Top Lineup {i + 1}")
                                st.dataframe(lineup)
                                st.metric("Total Salary", f"${salary:,.2f}")
                                st.metric("Projected Points", f"{points:.2f}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.markdown("""Please make sure your CSV file has the required columns:
                * Name
                * POS (F/M/D/GK)
                * DK Salary
                * Projection
                * DK Team""")

if __name__ == "__main__":
    main()