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

- **Base Implementation**: `round1.py`
- **Final Version**: `round1_FINAL.py`
- **GPT-Assisted Version**: `round1GPT.py`
- **Version 2**: `round1_v2.py`
- Features:
  - Basic trading strategies
  - Data visualization tools
  - Manual trading interface
  - Initial market analysis

### Round 2

- **Base Implementation**: `round2.py`
- **Final Version**: `round2_FIXED.py`
- **Version 1**: `round2_v1.py`
- **Version 2**: `round2_v2.py`
- **Version 3**: `round2_v3.py`
- **Version 4**: `round2_v4.py`
- **Shell Script**: `round2_sh.py`
- **Square Root Implementation**: `r2sq.py`
- Features:
  - Enhanced trading strategies
  - Improved data handling
  - Advanced market analysis
  - Performance optimization

### Round 3

- **Base Implementation**: `round3.py`
- **Final Version**: `round3_FINAL.py`
- **Version 1**: `round3_v1.py`
- **Version 2**: `round3_v2.py`
- **Version 3**: `round3_v3.py`
- **Version 4**: `round3_v4.py`
- Features:
  - Advanced trading strategies
  - Black-Scholes model implementation
  - Comprehensive data analysis
  - Risk management systems
  - Market making algorithms

### Round 4

- **Base Implementation**: `round4.py`
- **Final Version**: `round4_final.py`
- **Version 1**: `round4_v1.py`
- **Version 2**: `round4_v2.py`
- **Version 3**: `round4_v3.py`
- **Version 4**: `round4_v4.py`
- **Version 5**: `round4_v5.py`
- **Version 6**: `round4_v6.py`
- **Testing Version**: `round4_TEST.py`
- **R4 Implementation**: `r4.py`
- Features:
  - Sophisticated trading strategies
  - Advanced market making
  - Performance optimization
  - Comprehensive testing suite
  - Risk management systems

### Round 5

- **Base Implementation**: `round5.py`
- **Final Version**: `round5_Final.py`
- **Version 2**: `round5_v2.py`
- **Version 3**: `round5_v3.py`
- **Version 6**: `round5_v6.py`
- **Testing Version**: `round5_TEST.py`
- Features:
  - Latest trading strategies
  - Advanced market analysis
  - Comprehensive testing
  - Performance optimization
  - Risk management systems

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
