"""
Custom Credibility Scoring Engine
Allows users to customize weights for their own priorities.
Does NOT modify the ideal_score in the database.
"""
import pandas as pd
from typing import List, Dict, Optional


class CustomCredibilityScorer:
    """Calculate custom credibility scores with user-defined weights."""
    
    def __init__(self, custom_weights: Optional[Dict[str, int]] = None):
        # Default weights (same as original scoring)
        self.default_weights = {
            "processor_score": 8,
            "ram_gb": 7,
            "storage_gb": 6,
            "battery_mah": 5,
            "screen_inches": 4,
            "camera_mp": 4,
            "price_usd": 3,
            "weight_g": 2,
        }
        
        # Use custom weights if provided, otherwise use defaults
        self.score_cols_with_weights = custom_weights if custom_weights else self.default_weights.copy()
        
        # Lower is better for these columns
        self.smaller_is_better = ["price_usd", "weight_g"]
        
        # Quantile breakpoints
        self.quantiles = [0.25, 0.5, 0.75]
    
    def calculate_scores(self, products: List[Dict]) -> pd.DataFrame:
        """
        Calculate scores for a list of products using custom weights.
        
        Args:
            products: List of product dicts with spec keys
            
        Returns:
            DataFrame with products and calculated custom scores
        """
        if not products:
            return pd.DataFrame()
        
        df = pd.DataFrame(products)
        
        # Filter to only columns we can score
        available_score_cols = {k: v for k, v in self.score_cols_with_weights.items() 
                                if k in df.columns}
        
        if not available_score_cols:
            # No scoreable columns
            df['custom_score'] = None
            return df
        
        score_cols = list(available_score_cols.keys())
        
        # Convert to numeric and drop rows with ALL missing scoring data
        df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')
        
        # Only score products that have at least SOME data
        df_to_score = df[df[score_cols].notna().any(axis=1)].copy()
        
        if len(df_to_score) == 0:
            df['custom_score'] = None
            return df
        
        # Initialize score column
        df_to_score['custom_score'] = 0
        
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
                            df_to_score.at[idx, 'custom_score'] += weight
                    else:
                        if val >= q_val:
                            df_to_score.at[idx, 'custom_score'] += weight
        
        # Normalize to 0-100 scale
        max_score = df_to_score['custom_score'].max()
        if max_score > 0:
            df_to_score['custom_score'] = (df_to_score['custom_score'] / max_score) * 100
        
        # Merge scores back to original dataframe
        df.loc[df_to_score.index, 'custom_score'] = df_to_score['custom_score']
        
        return df


def calculate_custom_scores_from_db(db_session, custom_weights: Dict[str, int]) -> List[Dict]:
    """
    Calculate custom scores for all products in database with user-defined weights.
    
    Args:
        db_session: SQLAlchemy session
        custom_weights: Dict of metric names to weight values
        
    Returns:
        List of dicts with product info and custom scores
    """
    from app import Product
    
    # Get only products that have ideal_score (to match the original scoring dataset)
    products = db_session.query(Product).filter(Product.ideal_score.isnot(None)).all()
    
    product_dicts = [{
        'id': p.id,
        'product_name': p.product_name,
        'company_name': p.company_name,
        'category': p.category,
        'processor_score': p.processor_score,
        'ram_gb': p.ram_gb,
        'storage_gb': p.storage_gb,
        'battery_mah': p.battery_mah,
        'screen_inches': p.screen_inches,
        'camera_mp': p.camera_mp,
        'price_usd': p.price_usd,
        'weight_g': p.weight_g,
        'ideal_score': p.ideal_score,  # Keep original score for reference
        'qr_code_path': p.qr_code_path,
        'batch_number': p.batch_number,
    } for p in products]
    
    # Calculate custom scores
    scorer = CustomCredibilityScorer(custom_weights)
    scored_df = scorer.calculate_scores(product_dicts)
    
    # Convert back to list of dicts
    return scored_df.to_dict('records')
