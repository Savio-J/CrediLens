"""
Custom Credibility Scoring Engine
Allows users to customize weights for their own priorities.
Uses continuous percentile ranking for proportional scoring.
Does NOT modify the ideal_score in the database.
"""
import pandas as pd
from typing import List, Dict, Optional


class CustomCredibilityScorer:
    """Calculate custom credibility scores with user-defined weights using continuous scoring."""
    
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
    
    def calculate_scores(self, products: List[Dict]) -> pd.DataFrame:
        """
        Calculate scores for a list of products using custom weights and continuous percentile ranking.
        
        Each spec contributes: weight × percentile_rank (0.0 to 1.0)
        Final score normalized to 0-100 scale.
        
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
            df['custom_score'] = None
            return df
        
        score_cols = list(available_score_cols.keys())
        
        # Convert to numeric
        df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')
        
        # Only score products that have at least SOME data
        df_to_score = df[df[score_cols].notna().any(axis=1)].copy()
        
        if len(df_to_score) == 0:
            df['custom_score'] = None
            return df
        
        # Initialize score column
        df_to_score['custom_score'] = 0.0
        
        # Track actual weight used (in case some columns are missing)
        df_to_score['weight_used'] = 0.0
        
        # Calculate continuous percentile scores for each column
        for col, weight in available_score_cols.items():
            # Skip if column has no valid values
            if df_to_score[col].notna().sum() == 0:
                continue
            
            # Calculate percentile rank (0.0 to 1.0)
            # For "smaller is better" columns, invert the ranking
            if col in self.smaller_is_better:
                # Lower values get higher percentile (invert)
                percentiles = 1 - df_to_score[col].rank(pct=True, na_option='keep')
            else:
                # Higher values get higher percentile
                percentiles = df_to_score[col].rank(pct=True, na_option='keep')
            
            # Add weighted percentile to score (only for non-null values)
            mask = df_to_score[col].notna()
            df_to_score.loc[mask, 'custom_score'] += percentiles[mask] * weight
            df_to_score.loc[mask, 'weight_used'] += weight
        
        # Normalize to 0-100 scale based on weight used
        df_to_score['custom_score'] = (
            df_to_score['custom_score'] / df_to_score['weight_used'].clip(lower=1)
        ) * 100
        
        # Round to 2 decimal places
        df_to_score['custom_score'] = df_to_score['custom_score'].round(2)
        
        # Clean up temp column
        df_to_score = df_to_score.drop(columns=['weight_used'])
        
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
