# DraftKings Lineup Optimizer

## Overview
The DraftKings Lineup Optimizer is a Streamlit application designed to help users create optimal daily fantasy sports lineups for soccer. The application utilizes linear programming to maximize projected points while adhering to specific constraints such as salary cap and player positions.

## Features
- **Wide Layout**: The application is set to a wide layout for better visibility.
- **File Upload**: Users can upload CSV or Excel files containing player data.
- **Lineup Generation**: Generate optimal lineups based on player projections and constraints.
- **Position Constraints**: Ensure that lineups meet specific positional requirements (e.g., number of forwards, midfielders, defenders, and goalkeepers).
- **Randomization**: Generate multiple lineups with slight random adjustments to player projections for diversity.

## Technologies Used
- **Streamlit**: For building the web application.
- **Pandas**: For data manipulation and analysis.
- **Pulp**: For linear programming and optimization.
- **NumPy**: For numerical operations.
- **Altair**: For creating interactive visualizations.

## Installation
To run this application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DFS-Optimizer.git
   cd DFS-Optimizer
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Upload your CSV or Excel file containing the player pool with the required columns:
   - Name
   - POS (Position)
   - DK Salary
   - Projection
   - DK Team
2. Click the "Generate Top 3 Soccer Lineups" button to create optimal lineups.
3. Review the generated lineups, including total salary and projected points.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the Streamlit community for their support and resources.
- Special thanks to the developers of the libraries used in this project.

## Contact
For any inquiries or feedback, please reach out to [@loudogvideo on X).
