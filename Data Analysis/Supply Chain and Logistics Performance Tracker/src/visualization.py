"""
Visualization Engine for Supply Chain Analytics
Generates insight-driven visualizations including geospatial maps, bar charts, and trend lines.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os


class VisualizationEngine:
    """Creates visualizations for supply chain performance analysis."""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = 'visualizations'):
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def create_geospatial_map(self, title: str = 'Shipment Routes and Delay Hotspots') -> str:
        """
        Create a geospatial map showing shipment routes and delay hotspots.
        Uses Folium for interactive mapping.
        """
        try:
            import folium
        except ImportError:
            return "Folium not installed. Install with: pip install folium"
        
        # Coordinate mapping for origins and destinations
        coordinates = {
            'Shanghai': (31.2304, 121.4737),
            'Singapore': (1.3521, 103.8198),
            'Rotterdam': (51.9225, 4.4792),
            'Los Angeles': (34.0522, -118.2437),
            'Hamburg': (53.5511, 9.9937),
            'Dubai': (25.2048, 55.2708),
            'Tokyo': (35.6762, 139.6503),
            'New York': (40.7128, -74.0060),
            'London': (51.5074, -0.1278),
            'Busan': (35.1796, 129.0756),
            'Sydney': (-33.8688, 151.2093),
            'Toronto': (43.6532, -79.3832),
            'Mumbai': (19.0760, 72.8777)
        }
        
        # Calculate delay rates by route
        route_delays = self.df.groupby(['Origin', 'Destination']).agg({
            'Delay_Days': 'mean',
            'Shipment_ID': 'count'
        }).reset_index()
        route_delays.columns = ['Origin', 'Destination', 'Avg_Delay', 'Volume']
        
        # Create base map centered on average coordinates
        all_coords = [coordinates.get(loc, (0, 0)) for loc in coordinates.keys()]
        avg_lat = sum(c[0] for c in all_coords) / len(all_coords)
        avg_lon = sum(c[1] for c in all_coords) / len(all_coords)
        
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=3, tiles='cartodbpositron')
        
        # Add circles for origins (supply points)
        for origin in self.df['Origin'].unique():
            if origin in coordinates:
                lat, lon = coordinates[origin]
                vol = len(self.df[self.df['Origin'] == origin])
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=min(vol / 500, 20),
                    popup=f'{origin}: {vol} shipments',
                    color='#2E86AB',
                    fill=True,
                    fillColor='#2E86AB',
                    fillOpacity=0.6
                ).add_to(m)
        
        # Add circles for destinations (demand points)
        for dest in self.df['Destination'].unique():
            if dest in coordinates:
                lat, lon = coordinates[dest]
                vol = len(self.df[self.df['Destination'] == dest])
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=min(vol / 500, 15),
                    popup=f'{dest}: {vol} shipments',
                    color='#A23B72',
                    fill=True,
                    fillColor='#A23B72',
                    fillOpacity=0.6
                ).add_to(m)
        
        # Add heatmap-style markers for high-delay routes
        for _, row in route_delays.iterrows():
            if row['Origin'] in coordinates and row['Destination'] in coordinates:
                orig_coord = coordinates[row['Origin']]
                dest_coord = coordinates[row['Destination']]
                
                # Color based on delay severity
                if row['Avg_Delay'] > 5:
                    color = '#FF4136'  # Red for high delay
                elif row['Avg_Delay'] > 2:
                    color = '#FF851B'  # Orange for medium delay
                else:
                    color = '#2ECC40'  # Green for low delay
                
                # Draw line between origin and destination
                folium.PolyLine(
                    locations=[orig_coord, dest_coord],
                    color=color,
                    weight=1,
                    opacity=0.3,
                    popup=f"{row['Origin']} to {row['Destination']}: {row['Avg_Delay']:.1f} avg delay days"
                ).add_to(m)
        
        # Save map
        output_path = os.path.join(self.output_dir, 'shipment_routes_map.html')
        m.save(output_path)
        
        return output_path
    
    def create_carrier_penalty_chart(self, save: bool = True) -> Optional[str]:
        """
        Create a bar chart showing carrier SLA penalty recovery potential.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return "Matplotlib not installed"
        
        carrier_penalties = self.df.groupby('Carrier')['SLA_Penalty'].sum().reset_index()
        carrier_penalties.columns = ['Carrier', 'Total_Penalty']
        carrier_penalties = carrier_penalties.sort_values('Total_Penalty', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = ['#FF4136' if p > carrier_penalties['Total_Penalty'].mean() else '#2ECC40' 
                 for p in carrier_penalties['Total_Penalty']]
        
        bars = ax.barh(carrier_penalties['Carrier'], carrier_penalties['Total_Penalty'], color=colors)
        
        ax.set_xlabel('SLA Penalty Amount ($)', fontsize=12)
        ax.set_ylabel('Carrier', fontsize=12)
        ax.set_title('Carrier SLA Penalty Recovery Potential\n(Higher penalties indicate more recovery opportunity)', 
                    fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (carrier, penalty) in enumerate(zip(carrier_penalties['Carrier'], carrier_penalties['Total_Penalty'])):
            ax.text(penalty + carrier_penalties['Total_Penalty'].max() * 0.01, i, 
                   f'${penalty:,.0f}', va='center', fontsize=10)
        
        ax.axvline(x=carrier_penalties['Total_Penalty'].mean(), color='blue', linestyle='--', 
                  label=f'Average: ${carrier_penalties["Total_Penalty"].mean():,.0f}')
        ax.legend()
        
        plt.tight_layout()
        
        if save:
            output_path = os.path.join(self.output_dir, 'carrier_sla_penalties.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        
        plt.close()
        return None
    
    def create_monthly_delay_trend(self, save: bool = True) -> Optional[str]:
        """
        Create a trend line showing monthly delay rates over time.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return "Matplotlib not installed"
        
        monthly = self.df.groupby(['Year', 'Month']).agg({
            'Shipment_ID': 'count',
            'Delay_Days': lambda x: (x > 0).sum()
        }).reset_index()
        monthly.columns = ['Year', 'Month', 'Total_Shipments', 'Delayed_Shipments']
        monthly['Delay_Rate'] = monthly['Delayed_Shipments'] / monthly['Total_Shipments'] * 100
        
        # Create date string for plotting
        monthly['Date'] = monthly.apply(lambda r: f"{r['Year']}-{r['Month']:02d}", axis=1)
        
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Plot delay rate line
        ax1.plot(monthly['Date'], monthly['Delay_Rate'], marker='o', linewidth=2, 
                color='#FF4136', label='Delay Rate (%)')
        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Delay Rate (%)', fontsize=12, color='#FF4136')
        ax1.tick_params(axis='y', labelcolor='#FF4136')
        ax1.set_title('Monthly Shipment Delay Rate Trend', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add secondary axis for shipment volume
        ax2 = ax1.twinx()
        ax2.bar(monthly['Date'], monthly['Total_Shipments'], alpha=0.3, color='#2E86AB', 
               label='Total Shipments')
        ax2.set_ylabel('Total Shipments', fontsize=12, color='#2E86AB')
        ax2.tick_params(axis='y', labelcolor='#2E86AB')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        if save:
            output_path = os.path.join(self.output_dir, 'monthly_delay_trend.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        
        plt.close()
        return None
    
    def create_cargo_type_analysis(self, save: bool = True) -> Optional[str]:
        """Create visualization of delays by cargo type."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return "Matplotlib not installed"
        
        cargo_analysis = self.df.groupby('Cargo_Type').agg({
            'Shipment_ID': 'count',
            'Delay_Days': 'mean',
            'SLA_Penalty': 'sum'
        }).reset_index()
        cargo_analysis.columns = ['Cargo_Type', 'Shipments', 'Avg_Delay', 'Total_Penalty']
        cargo_analysis = cargo_analysis.sort_values('Avg_Delay', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left chart: Average delay by cargo type
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(cargo_analysis)))
        axes[0].bar(cargo_analysis['Cargo_Type'], cargo_analysis['Avg_Delay'], color=colors)
        axes[0].set_xlabel('Cargo Type', fontsize=12)
        axes[0].set_ylabel('Average Delay (Days)', fontsize=12)
        axes[0].set_title('Average Delay Days by Cargo Type', fontsize=14, fontweight='bold')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Right chart: Total SLA penalties by cargo type
        colors2 = plt.cm.Blues(np.linspace(0.3, 0.9, len(cargo_analysis)))
        axes[1].bar(cargo_analysis['Cargo_Type'], cargo_analysis['Total_Penalty'] / 1000, color=colors2)
        axes[1].set_xlabel('Cargo Type', fontsize=12)
        axes[1].set_ylabel('Total SLA Penalties ($ thousands)', fontsize=12)
        axes[1].set_title('Total SLA Penalties by Cargo Type', fontsize=14, fontweight='bold')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save:
            output_path = os.path.join(self.output_dir, 'cargo_type_analysis.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        
        plt.close()
        return None
    
    def create_all_visualizations(self) -> Dict[str, str]:
        """Generate all standard visualizations."""
        results = {}
        
        results['geospatial_map'] = self.create_geospatial_map()
        results['carrier_penalties'] = self.create_carrier_penalty_chart()
        results['delay_trend'] = self.create_monthly_delay_trend()
        results['cargo_analysis'] = self.create_cargo_type_analysis()
        
        return results
