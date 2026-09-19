"""
Report Generator Module
=======================

Generates insight-driven visualizations and reports highlighting
high-risk departments, unreliable source systems, and temporal trends
in flagged transactions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class ReportGenerator:
    """
    Generates comprehensive reports and visualizations for financial
    reconciliation analysis.
    
    This class creates:
    - Summary statistics reports
    - Department risk analysis charts
    - Source system reliability visualizations
    - Temporal trend analysis
    - Financial exposure reports
    
    Attributes:
        df (pd.DataFrame): Transaction DataFrame with anomaly flags.
        detector (AnomalyDetector): Anomaly detection engine instance.
        output_dir (Path): Directory for saving reports.
    """
    
    # Color palette for consistent styling
    COLOR_PALETTE = {
        "primary": "#2E86AB",
        "secondary": "#A23B72",
        "success": "#2ECC71",
        "warning": "#F39C12",
        "danger": "#E74C3C",
        "info": "#3498DB",
        "dark": "#2C3E50",
        "light": "#ECF0F1"
    }
    
    def __init__(self, df: pd.DataFrame, detector, output_dir: Path):
        """
        Initialize the Report Generator.
        
        Args:
            df: DataFrame containing transaction and anomaly data.
            detector: AnomalyDetector instance with analysis results.
            output_dir: Directory path for saving generated reports.
        """
        self.df = df.copy()
        self.detector = detector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure matplotlib style
        plt.style.use("seaborn-v0_8-whitegrid")
    
    def _save_figure(self, fig: plt.Figure, filename: str):
        """
        Save a matplotlib figure to the output directory.
        
        Args:
            fig: Matplotlib figure to save.
            filename: Name for the saved file.
        """
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    
    def generate_summary_report(self) -> Dict:
        """
        Generate a comprehensive summary report of the analysis.
        
        Returns:
            dict: Summary statistics and key findings.
        """
        df = self.df
        
        # Basic statistics
        total_transactions = len(df)
        total_expected = df["Expected_Amount"].sum()
        total_actual = df["Actual_Amount"].sum()
        total_discrepancy = df["Discrepancy"].abs().sum()
        avg_discrepancy_pct = df["Discrepancy_Pct"].abs().mean()
        
        # Status distribution
        status_dist = df["Status"].value_counts().to_dict() if "Status" in df.columns else {}
        
        # Anomaly counts - use columns that exist in df after detector processing
        anomaly_count = 0
        if "is_zscore_anomaly" in df.columns or "isolation_forest_flag" in df.columns:
            zscore_mask = df.get("is_zscore_anomaly", pd.Series(False, index=df.index))
            if_mask = df.get("isolation_forest_flag", pd.Series(False, index=df.index))
            anomaly_mask = zscore_mask | if_mask
            anomaly_count = int(anomaly_mask.sum())
        
        # Audit priority distribution
        priority_dist = {}
        if "audit_priority" in df.columns:
            priority_dist = df["audit_priority"].value_counts().to_dict()
        
        summary = {
            "report_title": "Financial Reconciliation Analysis Summary",
            "generated_at": datetime.now().isoformat(),
            "data_overview": {
                "total_transactions": total_transactions,
                "date_range": {
                    "start": str(df["Date"].min()),
                    "end": str(df["Date"].max())
                },
                "departments": df["Department"].nunique(),
                "source_systems": df["Source_System"].nunique()
            },
            "financial_summary": {
                "total_expected_amount": float(total_expected),
                "total_actual_amount": float(total_actual),
                "total_discrepancy": float(total_discrepancy),
                "average_discrepancy_percentage": float(avg_discrepancy_pct),
                "discrepancy_rate": float(total_discrepancy / total_expected * 100) if total_expected > 0 else 0
            },
            "anomaly_summary": {
                "total_anomalies_detected": int(anomaly_count),
                "anomaly_rate": float(anomaly_count / total_transactions * 100),
                "audit_priorities": priority_dist
            },
            "status_distribution": status_dist
        }
        
        # Save as text report
        self._save_summary_text(summary)
        
        return summary
    
    def _save_summary_text(self, summary: Dict):
        """
        Save summary report as a text file.
        
        Args:
            summary: Summary dictionary to save.
        """
        filepath = self.output_dir / "01_summary_report.txt"
        
        with open(filepath, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("FINANCIAL RECONCILIATION ANALYSIS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {summary['generated_at']}\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("DATA OVERVIEW\n")
            f.write("-" * 40 + "\n")
            overview = summary["data_overview"]
            f.write(f"Total Transactions: {overview['total_transactions']:,}\n")
            f.write(f"Date Range: {overview['date_range']['start']} to {overview['date_range']['end']}\n")
            f.write(f"Departments: {overview['departments']}\n")
            f.write(f"Source Systems: {overview['source_systems']}\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("FINANCIAL SUMMARY\n")
            f.write("-" * 40 + "\n")
            fin = summary["financial_summary"]
            f.write(f"Total Expected Amount: ${fin['total_expected_amount']:,.2f}\n")
            f.write(f"Total Actual Amount: ${fin['total_actual_amount']:,.2f}\n")
            f.write(f"Total Discrepancy: ${fin['total_discrepancy']:,.2f}\n")
            f.write(f"Average Discrepancy %: {fin['average_discrepancy_percentage']:.2f}%\n")
            f.write(f"Overall Discrepancy Rate: {fin['discrepancy_rate']:.2f}%\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("ANOMALY DETECTION RESULTS\n")
            f.write("-" * 40 + "\n")
            anom = summary["anomaly_summary"]
            f.write(f"Total Anomalies Detected: {anom['total_anomalies_detected']:,}\n")
            f.write(f"Anomaly Rate: {anom['anomaly_rate']:.2f}%\n")
            if anom["audit_priorities"]:
                f.write("\nAudit Priority Distribution:\n")
                for priority, count in anom["audit_priorities"].items():
                    f.write(f"  {priority}: {count:,}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")
    
    def generate_department_risk_analysis(self) -> plt.Figure:
        """
        Generate visualization of risk levels by department.
        
        Creates a horizontal bar chart showing departments ranked by
        high-priority anomaly count and average discrepancy percentage.
        
        Returns:
            plt.Figure: Generated figure (also saved to disk).
        """
        df = self.df
        
        # Aggregate by department
        dept_stats = df.groupby("Department").agg({
            "Transaction_ID": "count",
            "Discrepancy": ["sum", "mean"],
            "Discrepancy_Pct": "mean"
        }).reset_index()
        
        dept_stats.columns = [
            "Department", "Transaction_Count", 
            "Total_Discrepancy", "Avg_Discrepancy", "Avg_Discrepancy_Pct"
        ]
        
        # Add high priority count if available
        if "audit_priority" in df.columns:
            high_priority = df[df["audit_priority"] == "HIGH"].groupby("Department").size()
            dept_stats["High_Priority_Count"] = dept_stats["Department"].map(high_priority).fillna(0)
        else:
            dept_stats["High_Priority_Count"] = 0
        
        dept_stats = dept_stats.sort_values("High_Priority_Count", ascending=True)
        
        # Create figure with dual axes
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Bar chart for transaction count
        bars1 = ax1.barh(
            dept_stats["Department"],
            dept_stats["Transaction_Count"],
            color=self.COLOR_PALETTE["primary"],
            alpha=0.7,
            label="Total Transactions"
        )
        
        # Line plot for high priority count
        ax2 = ax1.twinx()
        line2 = ax2.plot(
            dept_stats["High_Priority_Count"],
            range(len(dept_stats)),
            color=self.COLOR_PALETTE["danger"],
            marker="o",
            linewidth=2,
            markersize=8,
            label="High Priority Anomalies"
        )
        
        # Styling
        ax1.set_xlabel("Transaction Count", fontsize=12)
        ax1.set_ylabel("Department", fontsize=12)
        ax2.set_xlabel("High Priority Count", fontsize=12)
        
        ax1.set_title(
            "Department Risk Analysis\n(Transaction Volume vs High Priority Anomalies)",
            fontsize=14,
            fontweight="bold"
        )
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
        
        ax1.grid(axis="x", alpha=0.3)
        
        self._save_figure(fig, "02_department_risk_analysis.png")
        
        return fig
    
    def generate_source_system_reliability(self) -> plt.Figure:
        """
        Generate visualization of source system reliability.
        
        Creates a scatter plot showing source systems by their
        discrepancy rate and transaction volume.
        
        Returns:
            plt.Figure: Generated figure (also saved to disk).
        """
        df = self.df
        
        # Aggregate by source system
        system_stats = df.groupby("Source_System").agg({
            "Transaction_ID": "count",
            "Discrepancy": "sum",
            "Expected_Amount": "sum",
            "Discrepancy_Pct": "mean"
        }).reset_index()
        
        system_stats.columns = [
            "Source_System", "Transaction_Count",
            "Total_Discrepancy", "Total_Expected", "Avg_Discrepancy_Pct"
        ]
        
        system_stats["Discrepancy_Rate"] = (
            system_stats["Total_Discrepancy"].abs() / system_stats["Total_Expected"] * 100
        )
        
        # Create bubble chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(
            system_stats["Transaction_Count"],
            system_stats["Avg_Discrepancy_Pct"],
            s=system_stats["Transaction_Count"] * 10,
            c=system_stats["Discrepancy_Rate"],
            cmap="RdYlGn_r",
            alpha=0.6,
            edgecolors="black",
            linewidth=1
        )
        
        # Add labels
        for idx, row in system_stats.iterrows():
            ax.annotate(
                row["Source_System"],
                (row["Transaction_Count"], row["Avg_Discrepancy_Pct"]),
                fontsize=9,
                ha="center",
                va="bottom"
            )
        
        # Styling
        ax.set_xlabel("Transaction Count", fontsize=12)
        ax.set_ylabel("Average Discrepancy %", fontsize=12)
        ax.set_title(
            "Source System Reliability Analysis\n(Bubble size = Transaction volume, Color = Discrepancy rate)",
            fontsize=14,
            fontweight="bold"
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Discrepancy Rate (%)", fontsize=10)
        
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        
        self._save_figure(fig, "03_source_system_reliability.png")
        
        return fig
    
    def generate_temporal_trends(self) -> plt.Figure:
        """
        Generate visualization of temporal trends in flagged transactions.
        
        Creates a time series showing monthly trends in anomalies
        and discrepancy amounts.
        
        Returns:
            plt.Figure: Generated figure (also saved to disk).
        """
        df = self.df.copy()
        
        # Ensure Date is datetime
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Create month column
        df["Year_Month"] = df["Date"].dt.to_period("M").astype(str)
        
        # Aggregate by month
        monthly_stats = df.groupby("Year_Month").agg({
            "Transaction_ID": "count",
            "Discrepancy": "sum",
            "Discrepancy_Pct": "mean"
        }).reset_index()
        
        monthly_stats.columns = ["Year_Month", "Transaction_Count", "Total_Discrepancy", "Avg_Discrepancy_Pct"]
        
        # Add anomaly count if available
        if "is_zscore_anomaly" in df.columns or "isolation_forest_flag" in df.columns:
            df["is_anomaly"] = df.get("is_zscore_anomaly", False) | df.get("isolation_forest_flag", False)
            anomaly_monthly = df[df["is_anomaly"]].groupby("Year_Month").size()
            monthly_stats["Anomaly_Count"] = monthly_stats["Year_Month"].map(anomaly_monthly).fillna(0)
        else:
            # Use high discrepancy as proxy
            monthly_stats["Anomaly_Count"] = df[df["Discrepancy_Pct"].abs() > 5].groupby("Year_Month").size()
            monthly_stats["Anomaly_Count"] = monthly_stats["Year_Month"].map(
                df[df["Discrepancy_Pct"].abs() > 5].groupby("Year_Month").size()
            ).fillna(0)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Top plot: Transaction and anomaly counts
        x = range(len(monthly_stats))
        ax1.bar(x, monthly_stats["Transaction_Count"], 
               color=self.COLOR_PALETTE["primary"], alpha=0.5, label="Total Transactions")
        ax1.plot(x, monthly_stats["Anomaly_Count"], 
                color=self.COLOR_PALETTE["danger"], marker="o", 
                linewidth=2, markersize=6, label="Flagged Anomalies")
        
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title("Monthly Transaction Volume and Anomaly Trends", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Discrepancy trends
        ax2.fill_between(x, monthly_stats["Total_Discrepancy"].abs(), 
                        color=self.COLOR_PALETTE["warning"], alpha=0.5, label="Total Discrepancy")
        ax2.plot(x, monthly_stats["Avg_Discrepancy_Pct"].abs(), 
                color=self.COLOR_PALETTE["secondary"], marker="s", 
                linewidth=2, markersize=6, label="Avg Discrepancy %")
        
        ax2.set_xlabel("Month", fontsize=12)
        ax2.set_ylabel("Discrepancy", fontsize=12)
        ax2.set_title("Monthly Discrepancy Trends", fontsize=14, fontweight="bold")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(x, monthly_stats["Year_Month"], rotation=45, ha="right")
        
        plt.tight_layout()
        self._save_figure(fig, "04_temporal_trends.png")
        
        return fig
    
    def generate_financial_exposure_report(self) -> Dict:
        """
        Generate detailed financial exposure report.
        
        Creates a comprehensive report of financial exposure by
        priority tier, department, and category.
        
        Returns:
            dict: Financial exposure metrics and top exposures.
        """
        df = self.df
        
        # Overall exposure
        total_exposure = df["Discrepancy"].abs().sum()
        
        # Exposure by audit priority
        if "audit_priority" in df.columns:
            priority_exposure = df.groupby("audit_priority").agg({
                "Transaction_ID": "count",
                "Discrepancy": ["sum", "mean", "std"]
            }).reset_index()
            priority_exposure.columns = [
                "Priority", "Count", "Total_Exposure", 
                "Avg_Exposure", "Std_Exposure"
            ]
        else:
            priority_exposure = None
        
        # Exposure by department
        dept_exposure = df.groupby("Department").agg({
            "Discrepancy": ["sum", "mean", "count"]
        }).reset_index()
        dept_exposure.columns = ["Department", "Total_Exposure", "Avg_Exposure", "Count"]
        dept_exposure = dept_exposure.sort_values("Total_Exposure", ascending=False)
        
        # Exposure by category
        if "Category" in df.columns:
            cat_exposure = df.groupby("Category").agg({
                "Discrepancy": ["sum", "mean", "count"]
            }).reset_index()
            cat_exposure.columns = ["Category", "Total_Exposure", "Avg_Exposure", "Count"]
            cat_exposure = cat_exposure.sort_values("Total_Exposure", ascending=False)
        else:
            cat_exposure = None
        
        # Top 20 highest exposure transactions
        top_exposures = df.nlargest(20, "Discrepancy")[
            ["Transaction_ID", "Date", "Department", "Category", 
             "Expected_Amount", "Actual_Amount", "Discrepancy", "Discrepancy_Pct"]
        ].copy()
        
        # Create exposure summary figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top left: Exposure by priority (if available)
        if priority_exposure is not None and len(priority_exposure) > 0:
            priorities = priority_exposure["Priority"].values
            exposures = priority_exposure["Total_Exposure"].abs().values
            colors = [self.COLOR_PALETTE.get(p.lower(), self.COLOR_PALETTE["primary"]) 
                     for p in priorities]
            axes[0, 0].bar(priorities, exposures, color=colors, alpha=0.7)
            axes[0, 0].set_title("Financial Exposure by Audit Priority", fontsize=14, fontweight="bold")
            axes[0, 0].set_ylabel("Total Exposure ($)")
            axes[0, 0].tick_params(axis="x", rotation=45)
        else:
            axes[0, 0].text(0.5, 0.5, "No priority data available", 
                          ha="center", va="center", transform=axes[0, 0].transAxes)
            axes[0, 0].set_title("Financial Exposure by Audit Priority", fontsize=14)
        
        # Top right: Top 10 departments by exposure
        top_depts = dept_exposure.head(10)
        axes[0, 1].barh(top_depts["Department"], top_depts["Total_Exposure"].abs(), 
                       color=self.COLOR_PALETTE["secondary"], alpha=0.7)
        axes[0, 1].set_title("Top 10 Departments by Financial Exposure", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Total Exposure ($)")
        axes[0, 1].invert_yaxis()
        
        # Bottom left: Exposure by category (if available)
        if cat_exposure is not None and len(cat_exposure) > 0:
            top_cats = cat_exposure.head(8)
            axes[1, 0].pie(top_cats["Total_Exposure"].abs(), 
                          labels=top_cats["Category"], autopct="%1.1f%%",
                          colors=plt.cm.Set3.colors[:len(top_cats)])
            axes[1, 0].set_title("Financial Exposure by Category", fontsize=14, fontweight="bold")
        else:
            axes[1, 0].text(0.5, 0.5, "No category data available", 
                          ha="center", va="center", transform=axes[1, 0].transAxes)
            axes[1, 0].set_title("Financial Exposure by Category", fontsize=14)
        
        # Bottom right: Discrepancy distribution
        axes[1, 1].hist(df["Discrepancy"].abs(), bins=50, 
                       color=self.COLOR_PALETTE["info"], alpha=0.7, edgecolor="black")
        axes[1, 1].set_title("Distribution of Absolute Discrepancies", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("Discrepancy Amount ($)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_xscale("log")
        
        plt.tight_layout()
        self._save_figure(fig, "05_financial_exposure.png")
        
        # Compile report dictionary
        report = {
            "total_financial_exposure": float(total_exposure),
            "exposure_by_priority": priority_exposure.to_dict("records") if priority_exposure is not None else None,
            "exposure_by_department": dept_exposure.to_dict("records"),
            "exposure_by_category": cat_exposure.to_dict("records") if cat_exposure is not None else None,
            "top_20_exposures": top_exposures.to_dict("records"),
            "report_generated_at": datetime.now().isoformat()
        }
        
        # Save as CSV
        top_exposures.to_csv(self.output_dir / "top_exposure_transactions.csv", index=False)
        
        return report
    
    def generate_all_reports(self) -> Dict:
        """
        Generate all available reports and visualizations.
        
        Returns:
            dict: Combined results from all report generation methods.
        """
        print("Generating summary report...")
        summary = self.generate_summary_report()
        
        print("Generating department risk analysis...")
        self.generate_department_risk_analysis()
        
        print("Generating source system reliability...")
        self.generate_source_system_reliability()
        
        print("Generating temporal trends...")
        self.generate_temporal_trends()
        
        print("Generating financial exposure report...")
        exposure = self.generate_financial_exposure_report()
        
        return {
            "summary": summary,
            "exposure_report": exposure,
            "output_directory": str(self.output_dir)
        }
