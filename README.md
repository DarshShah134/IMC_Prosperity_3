# IMC Prosperity 3 Trading Competition

This repository contains our implementation for the IMC Prosperity 3 trading competition. The competition involves developing algorithmic trading strategies for various market scenarios across multiple rounds.

## Project Structure

```
.
├── round1/              # Round 1 implementation
├── round2/              # Round 2 implementation
├── round3/              # Round 3 implementation
├── round4/              # Round 4 implementation
├── round5/              # Round 5 implementation
├── TESTER/              # Testing environment and utilities
├── backtests/           # Backtesting results and analysis
├── datamodel.py         # Common data model for all rounds
└── example-program.py   # Example implementation
```

## Implementation Details

### Round 1

- Initial implementation focusing on basic trading strategies
- Includes multiple algorithm versions (v0 through v3)
- Data visualization and manual trading tools

### Round 2

- Enhanced trading strategies
- Multiple algorithm versions (v0 through v2)
- Improved data handling and analysis

### Round 3

- Advanced trading strategies
- Multiple algorithm versions (v0 through v4)
- Includes Black-Scholes model implementation
- Comprehensive data analysis tools

### Round 4

- Sophisticated trading strategies
- Multiple algorithm versions (v1 through v6)
- Final implementation with optimized performance
- Includes testing and validation tools

### Round 5

- Latest round implementation
- Multiple algorithm versions (v2, v3, v6)
- Final implementation with comprehensive testing
- Advanced data modeling and analysis

## Development Environment

- Python 3.13
- Required packages:
  - numpy
  - pandas
  - (Additional dependencies listed in requirements.txt)

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/DarshShah134/IMC_Prosperity_3.git
   ```

2. Set up the virtual environment:

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the desired algorithm:
   ```bash
   python round5/round5_Final.py
   ```

## Testing

The `TESTER` directory contains tools for testing and validating the algorithms:

- Example algorithms
- Practice implementations
- Data visualization tools
- Manual trading interface

## Backtesting

The `backtests` directory contains:

- Backtesting results
- Performance analysis
- Strategy optimization data

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IMC Trading for organizing the Prosperity 3 competition
- Team members for their contributions
- Open source community for various tools and libraries used
