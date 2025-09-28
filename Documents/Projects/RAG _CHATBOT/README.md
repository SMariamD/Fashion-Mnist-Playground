# Auction House Inventory Management System

A comprehensive Java-based inventory management system designed for auction houses to manage collectibles, track items, and generate statistical reports.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Development Stages](#development-stages)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The Auction House Inventory Management System is a Java application that provides comprehensive management capabilities for auction houses dealing with various types of collectibles including books, cars, jewelry, and paintings. The system offers data validation, statistical analysis, and a user-friendly graphical interface.

## ✨ Features

### Core Functionality
- **Multi-Type Collectible Management**: Support for books, cars, jewelry, and paintings
- **CSV Data Import/Export**: Bulk data operations with error handling
- **Statistical Analysis**: Comprehensive reporting and analytics
- **Data Validation**: Robust input validation and error handling
- **Graphical User Interface**: Intuitive GUI for easy management
- **Search and Filter**: Advanced search capabilities across collectibles

### Key Capabilities
- Add, edit, and remove collectible items
- Sort items by various criteria (price, year, ID)
- Generate detailed statistics and reports
- Import data from CSV files with validation
- Export data and statistics
- Year estimation and range handling
- Condition validation and tracking

## 🏗️ System Architecture

### Class Hierarchy
```
Collectible (Base Class)
├── Book
├── Car
├── Jewellery
└── Painting

Supporting Classes:
├── CollectibleCollection
├── Manager
├── YearEstimate
├── CustomFrame
└── CollectibleComparators
```

### Core Components
- **CollectibleCollection**: Manages collections of various collectible types
- **Manager**: Handles UI interactions and business logic
- **CustomFrame**: Provides the graphical user interface
- **YearEstimate**: Handles year-related data and calculations

## 🚀 Installation

### Prerequisites
- Java Development Kit (JDK) 8 or higher
- Java IDE (Eclipse, IntelliJ IDEA, or VS Code)
- Git (for version control)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/SMariamD/Auction-House-Inventory-Management-System.git
   cd Auction-House-Inventory-Management-System
   ```

2. **Navigate to the project directory**
   ```bash
   cd AuctionHouse
   ```

3. **Compile the project**
   ```bash
   javac -cp src src/*.java
   ```

4. **Run the application**
   ```bash
   java -cp src Main
   ```

## 💻 Usage

### Starting the Application
1. Run the `Main.java` class
2. The GUI will launch automatically
3. Use the interface to manage your collectible inventory

### Adding Collectibles
1. Click "Add Item" button
2. Fill in the required information
3. Select the collectible type
4. Click "Save" to add to inventory

### Importing Data
1. Prepare CSV files with collectible data
2. Use the import functionality in the GUI
3. System will validate and process the data
4. Review any error messages for data issues

### Generating Reports
1. Click "Generate Stats" button
2. View comprehensive statistics
3. Export reports as needed

## 📁 Project Structure

```
Auction-House-Inventory-Management-System/
├── AuctionHouse/                    # Main application source code
│   ├── src/                        # Java source files
│   │   ├── Book.java               # Book collectible class
│   │   ├── Car.java                # Car collectible class
│   │   ├── Collectible.java        # Base collectible class
│   │   ├── CollectibleCollection.java # Collection management
│   │   ├── CollectibleComparators.java # Sorting comparators
│   │   ├── CustomFrame.java        # GUI main frame
│   │   ├── Jewellery.java          # Jewelry collectible class
│   │   ├── Main.java               # Application entry point
│   │   ├── Manager.java            # Business logic controller
│   │   ├── Painting.java           # Painting collectible class
│   │   ├── YearEstimate.java       # Year estimation utility
│   │   ├── Resources/              # Data files and resources
│   │   │   ├── collectibles.csv    # Sample collectible data
│   │   │   ├── invalidnumbers.csv  # Test data for validation
│   │   │   ├── missingfields.csv   # Test data for error handling
│   │   │   ├── statistics_summary.txt # Generated statistics
│   │   │   └── unexpectedstrings.csv # Test data for parsing
│   │   └── test/                   # Unit test files
│   │       ├── CollectibleCollectionTest.java
│   │       ├── CustomFrameTest.java
│   │       └── Resources/           # Test data files
│   └── Stage_1_Diagram.drawio.png  # UML diagram
├── Files for Stage 2/              # Stage 2 development files
│   ├── books.csv                   # Sample book data
│   ├── invalidnumbers.csv          # Test data
│   ├── missingfields.csv           # Test data
│   └── unexpectedstrings.csv       # Test data
├── Stage UML Diagrams/             # UML documentation
│   ├── Method Activity Diagram.drawio.png
│   ├── Stage 3 Diagram.drawio updated.drawio.png
│   ├── Stage 4 Diagram.drawio updated.drawio (1).drawio.png
│   ├── Stage_1_Diagram.drawio.png
│   └── Stage_2_Diagram.drawio.png
├── full ui window.PNG              # Application screenshot
├── Stage_1_Diagram.drawio.png      # Main UML diagram
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore rules
```

## 🔄 Development Stages

### Stage 1: Basic Structure
- Implemented Book class with core attributes
- Created BookCollection for managing books
- Developed basic CRUD operations
- Added UML class diagrams

### Stage 2: CSV Integration
- Implemented CSV file reading functionality
- Added data validation and error handling
- Created statistical analysis methods
- Enhanced error resilience

### Stage 3: Multi-Type Support
- Extended to support multiple collectible types
- Implemented YearEstimate class
- Added advanced statistical calculations
- Enhanced data validation

### Stage 4: GUI Implementation
- Developed graphical user interface
- Integrated Manager class for UI control
- Added advanced button functionalities
- Implemented data persistence

## 🧪 Testing

### Running Tests
```bash
# Navigate to test directory
cd AuctionHouse/src/test

# Compile test files
javac -cp ../..:../../junit-4.13.2.jar *.java

# Run tests
java -cp .:../../junit-4.13.2.jar:../../hamcrest-core-1.3.jar org.junit.runner.JUnitCore CollectibleCollectionTest
```

### Test Coverage
- Unit tests for CollectibleCollection
- GUI component testing
- Data validation testing
- CSV parsing validation
- Statistical calculation verification

## 📊 Data Format

### CSV File Structure
```csv
ID,Type,Title,Author,Year,Price,Condition,Owner
1,Book,The Great Gatsby,F. Scott Fitzgerald,1925,150.00,Good,John Doe
2,Car,Ford Mustang,Henry Ford,1965,25000.00,Excellent,Jane Smith
```

### Supported Collectible Types
- **Books**: Title, Author, Edition, Genre, Year, Price, Condition
- **Cars**: Make, Model, Year, Price, Condition, Mileage
- **Jewelry**: Type, Material, Year, Price, Condition, Weight
- **Paintings**: Artist, Title, Year, Price, Condition, Medium

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is part of academic coursework and is intended for educational purposes.

## 👥 Author

**Syeda Mariam Danish**
- GitHub: [@SMariamD](https://github.com/SMariamD)

## 📞 Support

For support or questions, please open an issue in the GitHub repository.

---

## 🔗 Related Links

- [Project Repository](https://github.com/SMariamD/Auction-House-Inventory-Management-System)
- [Issues](https://github.com/SMariamD/Auction-House-Inventory-Management-System/issues)
- [Pull Requests](https://github.com/SMariamD/Auction-House-Inventory-Management-System/pulls)

---

*This project was developed as part of F20-21SF Coursework and demonstrates advanced Java programming concepts, GUI development, and software engineering practices.*
