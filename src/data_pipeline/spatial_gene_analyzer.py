"""
Data pipeline component for analyzing spatial gene expression patterns.
"""

import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_pipeline.base_pipeline import BaseDataPipeline

logger = logging.getLogger(__name__)

class SpatialGeneAnalyzer(BaseDataPipeline):
    """Analyzes spatial patterns in gene expression data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the spatial gene analyzer.
        
        Args:
            config: Configuration dictionary containing analysis settings
        """
        super().__init__(config)
        self.input_dir = self.get_config_value('data.input_dir', 'data/input/')
        self.processed_dir = self.get_config_value('data.processed_dir', 'data/processed/')
        self.min_total_reads = self.get_config_value('gene_expression.min_total_reads', 1000)
        self.min_expression_percent = self.get_config_value('gene_expression.min_expression_percent', 0.10)
        self.top_genes_to_predict = self.get_config_value('gene_expression.top_genes_to_predict', 250)
        
    def execute(self) -> Dict[str, Any]:
        """
        Execute the spatial gene analysis process.
        
        Returns:
            Dictionary containing analysis results
        """
        self.log_info("Starting spatial gene expression analysis")
        
        try:
            # Ensure processed directory exists
            self._ensure_directory_exists(self.processed_dir)
            
            # Perform analysis
            results = self._analyze_spatial_patterns()
            
            if results:
                self.log_info("Spatial gene analysis completed successfully")
                return results
            else:
                self.log_error("Spatial gene analysis failed")
                return {}
                
        except Exception as e:
            self.log_error(f"Error during spatial gene analysis: {str(e)}")
            return {}
    
    def _analyze_spatial_patterns(self) -> Dict[str, Any]:
        """
        Analyze spatial patterns in gene expression data.
        
        Returns:
            Dictionary containing analysis results
        """
        try:
            input_path = self._resolve_path(self.input_dir)
            
            # Load gene expression data
            gene_expression_df = self._load_gene_expression_data(input_path)
            if gene_expression_df is None:
                return {}
                
            # Load spatial coordinates
            coordinates_df = self._load_spatial_coordinates(input_path)
            if coordinates_df is None:
                return {}
                
            # Filter genes based on expression thresholds
            filtered_genes = self._filter_genes_by_expression(gene_expression_df)
            
            # Identify spatially variable genes
            spatial_genes = self._identify_spatially_variable_genes(
                gene_expression_df[filtered_genes], coordinates_df
            )
            
            # Select top genes for prediction
            top_genes = self._select_top_prediction_genes(spatial_genes)
            
            # Perform dimensionality reduction for visualization
            reduced_data = self._perform_dimensionality_reduction(
                gene_expression_df[top_genes], coordinates_df
            )
            
            # Save analysis results
            results = {
                'filtered_genes': filtered_genes,
                'spatial_genes': spatial_genes,
                'top_prediction_genes': top_genes,
                'reduced_data_shape': reduced_data.shape if reduced_data is not None else None
            }
            
            self._save_analysis_results(results)
            
            return results
            
        except Exception as e:
            self.log_error(f"Error analyzing spatial patterns: {str(e)}")
            return {}
            
    def _load_gene_expression_data(self, input_path: Path) -> pd.DataFrame:
        """
        Load gene expression data from CSV file.
        
        Args:
            input_path: Path to input directory
            
        Returns:
            Gene expression DataFrame or None if failed
        """
        try:
            gene_file = input_path / "gene_expression.csv"
            if not gene_file.exists():
                self.log_warning("Gene expression file not found")
                return None
                
            df = pd.read_csv(gene_file, index_col=0)
            self.log_info(f"Loaded gene expression data: {df.shape[0]} spots, {df.shape[1]} genes")
            return df
            
        except Exception as e:
            self.log_error(f"Error loading gene expression data: {str(e)}")
            return None
            
    def _load_spatial_coordinates(self, input_path: Path) -> pd.DataFrame:
        """
        Load spatial coordinates from CSV file.
        
        Args:
            input_path: Path to input directory
            
        Returns:
            Coordinates DataFrame or None if failed
        """
        try:
            coord_file = input_path / "spot_coordinates.csv"
            if not coord_file.exists():
                self.log_warning("Spot coordinates file not found")
                return None
                
            df = pd.read_csv(coord_file, index_col=0)
            self.log_info(f"Loaded spatial coordinates: {df.shape[0]} spots")
            return df
            
        except Exception as e:
            self.log_error(f"Error loading spatial coordinates: {str(e)}")
            return None
            
    def _filter_genes_by_expression(self, gene_expression_df: pd.DataFrame) -> list:
        """
        Filter genes based on expression thresholds.
        
        Args:
            gene_expression_df: Gene expression DataFrame
            
        Returns:
            List of filtered gene names
        """
        # Calculate total reads per gene
        total_reads = gene_expression_df.sum(axis=0)
        
        # Calculate percentage of spots expressing each gene
        expressed_spots = (gene_expression_df > 0).sum(axis=0)
        expression_percent = expressed_spots / len(gene_expression_df)
        
        # Apply filters
        min_reads_mask = total_reads >= self.min_total_reads
        min_expr_mask = expression_percent >= self.min_expression_percent
        
        filtered_genes = gene_expression_df.columns[min_reads_mask & min_expr_mask].tolist()
        
        self.log_info(f"Filtered genes: {len(filtered_genes)} passed filters "
                     f"(total reads ≥ {self.min_total_reads}, "
                     f"expression ≥ {self.min_expression_percent*100:.1f}%)")
        
        return filtered_genes
        
    def _identify_spatially_variable_genes(self, gene_expression_df: pd.DataFrame, 
                                         coordinates_df: pd.DataFrame) -> list:
        """
        Identify genes with spatial expression patterns.
        
        Args:
            gene_expression_df: Filtered gene expression DataFrame
            coordinates_df: Spatial coordinates DataFrame
            
        Returns:
            List of spatially variable gene names
        """
        # Align indices
        common_spots = gene_expression_df.index.intersection(coordinates_df.index)
        expr_subset = gene_expression_df.loc[common_spots]
        coord_subset = coordinates_df.loc[common_spots]
        
        spatial_scores = {}
        
        # Simple Moran's I-like calculation for spatial autocorrelation
        for gene in expr_subset.columns:
            expression = expr_subset[gene].values
            if np.std(expression) > 0:  # Skip constant genes
                score = self._calculate_spatial_autocorrelation(expression, coord_subset)
                spatial_scores[gene] = score
                
        # Sort by spatial score and take top 50%
        sorted_genes = sorted(spatial_scores.items(), key=lambda x: x[1], reverse=True)
        num_spatial_genes = max(100, len(sorted_genes) // 2)  # At least 100, or half
        spatial_genes = [gene for gene, score in sorted_genes[:num_spatial_genes]]
        
        self.log_info(f"Identified {len(spatial_genes)} spatially variable genes")
        
        return spatial_genes
        
    def _calculate_spatial_autocorrelation(self, expression: np.array, 
                                         coordinates: pd.DataFrame) -> float:
        """
        Calculate spatial autocorrelation score for a gene.
        
        Args:
            expression: Gene expression values
            coordinates: Spatial coordinates
            
        Returns:
            Spatial autocorrelation score
        """
        # Simplified spatial autocorrelation calculation
        coords = coordinates[['x', 'y']].values
        n = len(expression)
        
        if n < 2:
            return 0.0
            
        # Calculate pairwise distances
        distances = np.sqrt(np.sum((coords[:, np.newaxis] - coords[np.newaxis, :])**2, axis=2))
        
        # Define neighbors as spots within 1.5x median distance
        median_dist = np.median(distances[distances > 0])
        neighbor_threshold = 1.5 * median_dist
        
        spatial_score = 0.0
        count = 0
        
        for i in range(n):
            neighbors = distances[i] < neighbor_threshold
            neighbors[i] = False  # Exclude self
            
            if np.any(neighbors):
                neighbor_mean = np.mean(expression[neighbors])
                spatial_score += abs(expression[i] - neighbor_mean)
                count += 1
                
        return spatial_score / max(count, 1) if count > 0 else 0.0
        
    def _select_top_prediction_genes(self, spatial_genes: list) -> list:
        """
        Select top genes for prediction based on variance.
        
        Args:
            spatial_genes: List of spatially variable genes
            
        Returns:
            List of top prediction genes
        """
        top_n = min(self.top_genes_to_predict, len(spatial_genes))
        
        # In a real implementation, this might use additional criteria
        # For now, we'll just take the first N genes
        selected_genes = spatial_genes[:top_n]
        
        self.log_info(f"Selected top {len(selected_genes)} genes for prediction")
        
        return selected_genes
        
    def _perform_dimensionality_reduction(self, gene_expression_df: pd.DataFrame, 
                                        coordinates_df: pd.DataFrame) -> np.array:
        """
        Perform PCA for dimensionality reduction and visualization.
        
        Args:
            gene_expression_df: Gene expression DataFrame
            coordinates_df: Coordinates DataFrame
            
        Returns:
            Reduced dimensionality data or None if failed
        """
        try:
            # Align indices
            common_spots = gene_expression_df.index.intersection(coordinates_df.index)
            expr_subset = gene_expression_df.loc[common_spots]
            
            if len(common_spots) < 10:  # Need sufficient samples
                return None
                
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(expr_subset)
            
            # Apply PCA
            pca = PCA(n_components=min(50, len(common_spots), len(expr_subset.columns)))
            reduced_data = pca.fit_transform(scaled_data)
            
            self.log_info(f"PCA completed: {reduced_data.shape[1]} components "
                         f"explain {pca.explained_variance_ratio_.sum():.2%} variance")
            
            return reduced_data
            
        except Exception as e:
            self.log_warning(f"Error in dimensionality reduction: {str(e)}")
            return None
            
    def _save_analysis_results(self, results: Dict[str, Any]) -> None:
        """
        Save analysis results to processed directory.
        
        Args:
            results: Analysis results dictionary
        """
        try:
            processed_path = self._resolve_path(self.processed_dir)
            results_file = processed_path / "spatial_analysis_results.npz"
            
            # Save key results
            np.savez_compressed(
                results_file,
                filtered_genes=results.get('filtered_genes', []),
                spatial_genes=results.get('spatial_genes', []),
                top_prediction_genes=results.get('top_prediction_genes', []),
                metadata={
                    'min_total_reads': self.min_total_reads,
                    'min_expression_percent': self.min_expression_percent,
                    'top_genes_to_predict': self.top_genes_to_predict,
                    'analysis_timestamp': pd.Timestamp.now().isoformat()
                }
            )
            
            self.log_info(f"Saved analysis results to {results_file}")
            
        except Exception as e:
            self.log_warning(f"Could not save analysis results: {str(e)}")
            
    def analyze_spatial_patterns(self) -> Dict[str, Any]:
        """
        Public method to analyze spatial patterns.
        
        Returns:
            Dictionary containing analysis results
        """
        return self.execute()

__all__ = ['SpatialGeneAnalyzer']
