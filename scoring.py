"""
Credibility Scoring Engine
Calculates ideal scores for products based on weighted quantile analysis.
"""
import pandas as pd
from typing import List, Dict, Optional


class CredibilityScorer:
    """Calculate credibility scores for electronics products."""
    
    def __init__(self):
        # Scoring configuration
        self.score_cols_with_weights = {
            "processor_score": 8,
            "ram_gb": 7,
            "storage_gb": 6,
            "battery_mah": 5,
            "screen_inches": 4,
            "camera_mp": 4,
            "price_usd": 3,
            "weight_g": 2,
        }
        
        # Lower is better for these columns
        self.smaller_is_better = ["price_usd", "weight_g"]
        
        # Quantile breakpoints
        self.quantiles = [0.25, 0.5, 0.75]
    
    def calculate_scores(self, products: List[Dict]) -> pd.DataFrame:
        """
        Calculate scores for a list of products.
        
        Args:
            products: List of product dicts with spec keys
            
        Returns:
            DataFrame with products and calculated scores
        """
        if not products:
            return pd.DataFrame()
        
        df = pd.DataFrame(products)
        
        # Filter to only columns we can score
        available_score_cols = {k: v for k, v in self.score_cols_with_weights.items() 
                                if k in df.columns}
        
        if not available_score_cols:
            # No scoreable columns
            df['ideal_score'] = None
            return df
        
        score_cols = list(available_score_cols.keys())
        
        # Convert to numeric and drop rows with ALL missing scoring data
        df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')
        
        # Only score products that have at least SOME data
        df_to_score = df[df[score_cols].notna().any(axis=1)].copy()
        
        if len(df_to_score) == 0:
            df['ideal_score'] = None
            return df
        
        # Initialize score column
        df_to_score['ideal_score'] = 0
        
        # Calculate quantiles (only on non-null values)
        quantiles_df = df_to_score[score_cols].quantile(self.quantiles)
        
        # Score each product
        for idx in df_to_score.index:
            for quantile in self.quantiles:
                for col, weight in available_score_cols.items():
                    val = df_to_score.at[idx, col]
                    
                    # Skip if value is missing
                    if pd.isna(val):
                        continue
                    
                    q_val = quantiles_df.at[quantile, col]
                    
                    # Skip if quantile is NaN
                    if pd.isna(q_val):
                        continue
                    
                    # Score based on whether smaller is better
                    if col in self.smaller_is_better:
                        if val <= q_val:
                            df_to_score.at[idx, 'ideal_score'] += weight
                    else:
                        if val >= q_val:
                            df_to_score.at[idx, 'ideal_score'] += weight
        
        # Normalize to 0-100 scale
        max_score = df_to_score['ideal_score'].max()
        if max_score > 0:
            df_to_score['ideal_score'] = (df_to_score['ideal_score'] / max_score) * 100
        
        # Merge scores back to original dataframe
        df.loc[df_to_score.index, 'ideal_score'] = df_to_score['ideal_score']
        
        return df
    
    def score_new_product(self, new_product: Dict, existing_products: List[Dict]) -> float:
        """
        Score a single new product against existing products.
        
        Args:
            new_product: Dict with product specs
            existing_products: List of existing product dicts
            
        Returns:
            Calculated ideal score (0-100)
        """
        # Combine new product with existing ones
        all_products = existing_products + [new_product]
        
        # Calculate scores for all
        scored_df = self.calculate_scores(all_products)
        
        # Return score for the new product (last row)
        if len(scored_df) > 0:
            score = scored_df.iloc[-1]['ideal_score']
            return round(float(score), 2) if pd.notna(score) else None
        
        return None


def score_product_from_db(product_dict: Dict, db_session) -> Optional[float]:
    """
    Helper function to score a product using all products from database.
    
    Args:
        product_dict: New product data as dict
        db_session: SQLAlchemy session
        
    Returns:
        Calculated score or None
    """
    from app import Product
    
    # Get all existing products from DB
    existing = db_session.query(Product).all()
    
    existing_dicts = [{
        'processor_score': p.processor_score,
        'ram_gb': p.ram_gb,
        'storage_gb': p.storage_gb,
        'battery_mah': p.battery_mah,
        'screen_inches': p.screen_inches,
        'camera_mp': p.camera_mp,
        'price_usd': p.price_usd,
        'weight_g': p.weight_g,
    } for p in existing]
    
    scorer = CredibilityScorer()
    return scorer.score_new_product(product_dict, existing_dicts)
