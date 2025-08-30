
# Check for required packages
import sys
required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("Missing required packages:", ', '.join(missing_packages))
    print("Install with: pip install pandas scikit-learn matplotlib seaborn")
    print("Or create virtual environment:")
    print("   python3 -m venv venv")
    print("   source venv/bin/activate")
    print("   pip install pandas scikit-learn matplotlib seaborn")
    sys.exit(1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Set matplotlib style (fallback if seaborn style not available)
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')

class TelanganaSupplyChainAnalysis:
    def __init__(self, csv_file):
        """Initialize the analysis with the dataset."""
        self.csv_file = csv_file
        self.df = None
        self.le_yard = LabelEncoder()
        self.le_comm = LabelEncoder()
        self.model = None
        self.X_test = None
        self.y_test = None
        
    def load_and_clean_data(self):
        """Load and clean the dataset."""
        print("=" * 60)
        print("LOADING AND CLEANING DATA")
        print("=" * 60)
        
        self.df = pd.read_csv(self.csv_file)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Convert date column - handle ISO format with timezone
        self.df['DDate'] = pd.to_datetime(self.df['DDate'], format='ISO8601')
        
        # Remove rows with missing values
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        print(f"Removed {initial_rows - len(self.df)} rows with missing values")
        
        # Remove zero/negative prices (data quality issues) - use 'Model' column name
        self.df = self.df[(self.df['Model'] > 0) & (self.df['Minimum'] > 0) & (self.df['Maximum'] > 0)]
        
        # Create derived features for analysis
        self.df['PriceRange'] = self.df['Maximum'] - self.df['Minimum']
        self.df['PriceVolatility'] = self.df['PriceRange'] / self.df['Model']
        self.df['DayOfWeek'] = self.df['DDate'].dt.dayofweek
        self.df['Month'] = self.df['DDate'].dt.month
        self.df['Day'] = self.df['DDate'].dt.day
        
        print(f"Final dataset: {self.df.shape[0]} rows after cleaning")
        print(f"Date range: {self.df['DDate'].min()} to {self.df['DDate'].max()}")
        
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA."""
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 60)
        
        # Basic statistics
        print("\n1. DATASET OVERVIEW:")
        print(f"Total records: {len(self.df)}")
        print(f"Unique yards: {self.df['YardName'].nunique()}")
        print(f"Unique commodities: {self.df['CommName'].nunique()}")
        print(f"Date range: {(self.df['DDate'].max() - self.df['DDate'].min()).days} days")
        
        print("\n2. PRICE STATISTICS:")
        price_stats = self.df[['Model', 'Minimum', 'Maximum', 'Arrivals']].describe()
        print(price_stats)
        
        print("\n3. TOP 10 COMMODITIES BY FREQUENCY:")
        top_commodities = self.df['CommName'].value_counts().head(10)
        print(top_commodities)
        
        print("\n4. TOP 10 YARDS BY ACTIVITY:")
        top_yards = self.df['YardName'].value_counts().head(10)
        print(top_yards)
        
        print("\n5. PRICE VOLATILITY ANALYSIS:")
        volatility_stats = self.df.groupby('CommName')['PriceVolatility'].agg(['mean', 'std']).round(3)
        volatility_stats = volatility_stats.sort_values('mean', ascending=False)
        print("Most volatile commodities (Top 10):")
        print(volatility_stats.head(10))
        
        return top_commodities, top_yards
        
    def create_visualizations(self, top_commodities, top_yards):
        """Create comprehensive visualizations."""
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Price trends over time for top commodities
        plt.subplot(2, 3, 1)
        top_5_commodities = top_commodities.head(5).index
        for commodity in top_5_commodities:
            comm_data = self.df[self.df['CommName'] == commodity].groupby('DDate')['Model'].mean()
            plt.plot(comm_data.index, comm_data.values, label=commodity, marker='o', markersize=2)
        plt.title('Price Trends - Top 5 Commodities', fontsize=12, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Model Price (Rs/Quintal)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. Average prices by commodity
        plt.subplot(2, 3, 2)
        avg_prices = self.df.groupby('CommName')['Model'].mean().sort_values(ascending=False).head(10)
        avg_prices.plot(kind='bar', color='skyblue')
        plt.title('Top 10 Commodities by Average Price', fontsize=12, fontweight='bold')
        plt.xlabel('Commodity')
        plt.ylabel('Average Model Price (Rs/Quintal)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 3. Market activity by yard
        plt.subplot(2, 3, 3)
        top_yards.head(10).plot(kind='bar', color='lightgreen')
        plt.title('Market Activity - Top 10 Yards', fontsize=12, fontweight='bold')
        plt.xlabel('Yard Name')
        plt.ylabel('Number of Records')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 4. Price distribution
        plt.subplot(2, 3, 4)
        plt.hist(self.df['Model'], bins=50, color='orange', alpha=0.7, edgecolor='black')
        plt.title('Distribution of Model Prices', fontsize=12, fontweight='bold')
        plt.xlabel('Model Price (Rs/Quintal)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 5. Seasonal patterns (day of week)
        plt.subplot(2, 3, 5)
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_pattern = self.df.groupby('DayOfWeek')['Model'].mean()
        plt.bar(range(7), weekly_pattern.values, color='purple', alpha=0.7)
        plt.title('Average Prices by Day of Week', fontsize=12, fontweight='bold')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Model Price (Rs/Quintal)')
        plt.xticks(range(7), day_names)
        plt.grid(True, alpha=0.3)
        
        # 6. Price volatility by commodity
        plt.subplot(2, 3, 6)
        volatility_data = self.df.groupby('CommName')['PriceVolatility'].mean().sort_values(ascending=False).head(10)
        volatility_data.plot(kind='bar', color='red', alpha=0.7)
        plt.title('Price Volatility - Top 10 Commodities', fontsize=12, fontweight='bold')
        plt.xlabel('Commodity')
        plt.ylabel('Average Price Volatility')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('telangana_market_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizations created and saved as 'telangana_market_analysis.png'")
        
    def prepare_ml_features(self):
        """Prepare features for machine learning."""
        print("\n" + "=" * 60)
        print("PREPARING MACHINE LEARNING FEATURES")
        print("=" * 60)
        
        # Create a working copy
        ml_df = self.df.copy()
        
        # Encode categorical variables
        # Label encoding for YardName and CommName (preserving the fitted encoders)
        ml_df['YardName_encoded'] = self.le_yard.fit_transform(ml_df['YardName'])
        ml_df['CommName_encoded'] = self.le_comm.fit_transform(ml_df['CommName'])
        
        # Feature engineering for time-based patterns
        ml_df['DayOfYear'] = ml_df['DDate'].dt.dayofyear
        ml_df['WeekOfYear'] = ml_df['DDate'].dt.isocalendar().week
        
        # Create lagged features (price from previous days)
        for commodity in ml_df['CommName'].unique():
            comm_mask = ml_df['CommName'] == commodity
            comm_data = ml_df[comm_mask].sort_values('DDate')
            
            # Calculate 3-day and 7-day moving averages
            ml_df.loc[comm_mask, 'Price_3day_avg'] = comm_data['Model'].rolling(window=3, min_periods=1).mean().values
            ml_df.loc[comm_mask, 'Price_7day_avg'] = comm_data['Model'].rolling(window=7, min_periods=1).mean().values
        
        # Fill any remaining NaN values
        ml_df['Price_3day_avg'] = ml_df['Price_3day_avg'].fillna(ml_df['Model'])
        ml_df['Price_7day_avg'] = ml_df['Price_7day_avg'].fillna(ml_df['Model'])
        
        # Select features for the model
        feature_columns = [
            'YardName_encoded', 'CommName_encoded', 'Arrivals',
            'DayOfWeek', 'Month', 'Day', 'DayOfYear', 'WeekOfYear',
            'Minimum', 'Maximum', 'PriceRange', 'PriceVolatility',
            'Price_3day_avg', 'Price_7day_avg'
        ]
        
        X = ml_df[feature_columns]
        y = ml_df['Model']  # Target variable: Model price
        
        print(f"Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        print(f"Feature columns: {feature_columns}")
        
        return X, y, ml_df
        
    def train_random_forest_model(self, X, y):
        """Train Random Forest model for price prediction."""
        print("\n" + "=" * 60)
        print("TRAINING RANDOM FOREST MODEL")
        print("=" * 60)
        
        # Split data into training and testing sets
        # Using 80% for training, 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        # Initialize Random Forest Regressor
        # Random Forest: Ensemble method that creates multiple decision trees
        # and averages their predictions for better accuracy and reduced overfitting
        self.model = RandomForestRegressor(
            n_estimators=100,     # Number of trees in the forest
            max_depth=15,         # Maximum depth of trees (prevents overfitting)
            min_samples_split=5,  # Minimum samples required to split a node
            min_samples_leaf=2,   # Minimum samples required at leaf node
            random_state=42,      # For reproducible results
            n_jobs=-1            # Use all available CPU cores
        )
        
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel Performance Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.2f} Rs/Quintal")
        print(f"Root Mean Square Error (RMSE): {rmse:.2f} Rs/Quintal")
        print(f"R² Score: {r2:.4f}")
        print(f"Model explains {r2*100:.2f}% of price variance")
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Store test data for recommendations
        self.X_test = X_test
        self.y_test = y_test
        
        return y_pred, feature_importance
        
    def generate_recommendations(self, ml_df):
        """Generate actionable buy/sell recommendations for farmers."""
        print("\n" + "=" * 60)
        print("GENERATING BUY/SELL RECOMMENDATIONS")
        print("=" * 60)
        
        # Current market analysis (last available date)
        latest_date = self.df['DDate'].max()
        current_data = self.df[self.df['DDate'] == latest_date].copy()
        
        # Calculate price trends (compare with 7-day average)
        recommendations = []
        
        for commodity in current_data['CommName'].unique():
            comm_data = self.df[self.df['CommName'] == commodity].copy()
            
            if len(comm_data) >= 7:  # Need at least 7 days of data
                # Current price vs 7-day average
                current_price = comm_data[comm_data['DDate'] == latest_date]['Model'].mean()
                week_avg = comm_data.tail(7)['Model'].mean()
                
                # Calculate price change percentage
                price_change = ((current_price - week_avg) / week_avg) * 100
                
                # Calculate volatility (risk indicator)
                volatility = comm_data['PriceVolatility'].mean()
                
                # Generate recommendation based on price trends and volatility
                if price_change < -5 and volatility < 0.2:  # Price dropped significantly, low risk
                    recommendation = "STRONG BUY"
                    reason = f"Price dropped {abs(price_change):.1f}% below weekly average, low volatility"
                elif price_change < -2:
                    recommendation = "BUY"
                    reason = f"Price {abs(price_change):.1f}% below weekly average"
                elif price_change > 5 and volatility < 0.3:
                    recommendation = "SELL"
                    reason = f"Price {price_change:.1f}% above weekly average, good selling opportunity"
                elif price_change > 10:
                    recommendation = "STRONG SELL"
                    reason = f"Price peaked {price_change:.1f}% above average"
                else:
                    recommendation = "HOLD"
                    reason = "Price stable, wait for better opportunity"
                
                recommendations.append({
                    'Commodity': commodity,
                    'Current_Price': current_price,
                    'Weekly_Avg': week_avg,
                    'Price_Change_%': price_change,
                    'Volatility': volatility,
                    'Recommendation': recommendation,
                    'Reason': reason
                })
        
        # Convert to DataFrame and sort by price change
        rec_df = pd.DataFrame(recommendations)
        rec_df = rec_df.sort_values('Price_Change_%', ascending=False)
        
        print(f"MARKET RECOMMENDATIONS FOR {latest_date.strftime('%B %d, %Y')}:")
        print("=" * 80)
        
        # Display top recommendations
        strong_buy = rec_df[rec_df['Recommendation'] == 'STRONG BUY']
        strong_sell = rec_df[rec_df['Recommendation'] == 'STRONG SELL']
        
        if len(strong_buy) > 0:
            print("\n🟢 STRONG BUY OPPORTUNITIES:")
            for _, row in strong_buy.head(5).iterrows():
                print(f"• {row['Commodity']}: Rs {row['Current_Price']:.0f}/quintal")
                print(f"  {row['Reason']}")
                print()
        
        if len(strong_sell) > 0:
            print("\n🔴 STRONG SELL OPPORTUNITIES:")
            for _, row in strong_sell.head(5).iterrows():
                print(f"• {row['Commodity']}: Rs {row['Current_Price']:.0f}/quintal")
                print(f"  {row['Reason']}")
                print()
        
        # Display all recommendations in tabular format
        print("\nCOMPLETE RECOMMENDATION TABLE:")
        print("-" * 120)
        print(f"{'Commodity':<20} {'Current':<8} {'Weekly':<8} {'Change%':<8} {'Volatility':<10} {'Action':<12} {'Reason'}")
        print("-" * 120)
        
        for _, row in rec_df.iterrows():
            print(f"{row['Commodity'][:19]:<20} {row['Current_Price']:<8.0f} {row['Weekly_Avg']:<8.0f} "
                  f"{row['Price_Change_%']:<8.1f} {row['Volatility']:<10.3f} "
                  f"{row['Recommendation']:<12} {row['Reason'][:40]}")
        
        # Save recommendations to CSV
        rec_df.to_csv('telangana_recommendations.csv', index=False)
        print(f"\n✓ Recommendations saved to 'telangana_recommendations.csv'")
        
        return rec_df
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("TELANGANA AGRICULTURAL SUPPLY CHAIN ANALYSIS")
        print("Micro-Supply Chain for Rural Producers Project")
        print("=" * 80)
        
        try:
            # Step 1: Load and clean data
            self.load_and_clean_data()
            
            # Step 2: Exploratory Data Analysis
            top_commodities, top_yards = self.exploratory_data_analysis()
            
            # Step 3: Create visualizations
            self.create_visualizations(top_commodities, top_yards)
            
            # Step 4: Prepare ML features
            X, y, ml_df = self.prepare_ml_features()
            
            # Step 5: Train Random Forest model
            y_pred, feature_importance = self.train_random_forest_model(X, y)
            
            # Step 6: Generate recommendations
            recommendations = self.generate_recommendations(ml_df)
            
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("Files generated:")
            print("   • telangana_market_analysis.png - Comprehensive visualizations")
            print("   • telangana_recommendations.csv - Buy/sell recommendations")
            print("=" * 80)
            
            return {
                'model': self.model,
                'recommendations': recommendations,
                'feature_importance': feature_importance,
                'data': self.df
            }
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise


def main():
    """Main function to run the analysis."""
    csv_file = "day_prices_between_01-07-2025_31-07-2025.csv"
    
    # Check if file exists
    try:
        analysis = TelanganaSupplyChainAnalysis(csv_file)
        results = analysis.run_complete_analysis()
        
        print("\nKEY INSIGHTS FOR FARMERS:")
        print("1. Use the price predictions to plan when to sell crops")
        print("2. Monitor volatile commodities for quick profit opportunities")
        print("3. Check weekly recommendations for optimal trading decisions")
        print("4. Consider seasonal patterns when planning crop cycles")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'")
        print("Please ensure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
