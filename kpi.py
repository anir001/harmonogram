import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
from typing import Dict, List
import matplotlib.pyplot as plt

class KPICalculator:
    def __init__(self, harmonogram_file: str, jobs_file: str, hours_per_day: int = 8):
        """Initialize KPICalculator with harmonogram and jobs CSV files."""
        self.harmonogram_file = harmonogram_file
        self.jobs_file = jobs_file
        self.hours_per_day = hours_per_day
        self.harmonogram_data = None
        self.jobs_data = None
        self.working_days = []
        self.kpi_results = {}

    def load_data(self):
        """Load data from harmonogram.csv and jobs.csv."""
        try:
            self.harmonogram_data = pd.read_csv(self.harmonogram_file, delimiter=';')
            if not all(col in self.harmonogram_data.columns for col in ['data', 'zadanie', 'czas', 'pracownik', 'uwagi']):
                raise ValueError("harmonogram.csv missing required columns")
        except Exception as e:
            raise ValueError(f"Error reading harmonogram.csv: {e}")

        try:
            self.jobs_data = pd.read_csv(self.jobs_file, delimiter=';')
            if not all(col in self.jobs_data.columns for col in ['job_id', 'deadline']):
                raise ValueError("jobs.csv missing required columns")
        except Exception as e:
            raise ValueError(f"Error reading jobs.csv: {e}")

        # Generate working days from earliest deadline to latest end_date
        harmonogram_dates = pd.to_datetime(self.harmonogram_data['data']).dt.date
        jobs_deadlines = pd.to_datetime(self.jobs_data['deadline']).dt.date
        if jobs_deadlines.empty:
            raise ValueError("No valid deadlines found in jobs.csv")
        start_date = min(min(harmonogram_dates), min(jobs_deadlines))
        end_date = max(max(harmonogram_dates), max(jobs_deadlines))
        self.working_days = self._generate_working_days(start_date, end_date)
        print(f"Working days for KPI calculation: {self.working_days}")

    def _generate_working_days(self, start_date, end_date) -> List[datetime.date]:
        """Generate list of working days (Monday-Friday) in the given period."""
        working_days = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday-Friday
                working_days.append(current_date)
            current_date += timedelta(days=1)
        return working_days

    def _count_all_days(self, start_date: datetime.date, end_date: datetime.date) -> int:
        """Count all days (including weekends) between start_date and end_date (inclusive)."""
        if start_date > end_date:
            return 0
        return (end_date - start_date).days

    def _count_working_days(self, start_date: datetime.date, end_date: datetime.date) -> int:
        """Count working days (Monday-Friday) between start_date and end_date (inclusive)."""
        if start_date > end_date:
            return 0
        return np.busday_count(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    def _get_day_index(self, date: datetime.date) -> int:
        """Return the index of the day in working_days list."""
        return self.working_days.index(date) if date in self.working_days else len(self.working_days) - 1

    def calculate_kpi(self) -> Dict:
        """Calculate KPIs based on harmonogram and jobs data."""
        if self.harmonogram_data is None or self.jobs_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Initialize KPI dictionary
        self.kpi_results = {
            'on_time_percentage': 0.0,
            'average_delay_days': 0.0,
            'on_time_hours_percentage': 0.0,
            'worker_utilization': {},
            'tasks_per_worker': {},
            'average_task_duration_days': 0.0
        }

        # 1. Procent zadań ukończonych w terminie
        tasks = self.harmonogram_data['zadanie'].unique()
        on_time_tasks = 0
        for task in tasks:
            task_rows = self.harmonogram_data[self.harmonogram_data['zadanie'] == task]
            is_late = any(task_rows['uwagi'] == 'Po terminie')
            if not is_late:
                on_time_tasks += 1
        self.kpi_results['on_time_percentage'] = (on_time_tasks / len(tasks) * 100) if len(tasks) > 0 else 0.0

        # 2. Średnie opóźnienie dla zadań po terminie (wszystkie dni)
        delays = []
        for task in tasks:
            task_rows = self.harmonogram_data[self.harmonogram_data['zadanie'] == task]
            if any(task_rows['uwagi'] == 'Po terminie'):
                end_date = pd.to_datetime(task_rows['data']).dt.date.max()
                deadline = pd.to_datetime(self.jobs_data[self.jobs_data['job_id'] == task]['deadline']).dt.date.iloc[0]
                if end_date > deadline:
                    delay_days = self._count_all_days(deadline, end_date)
                    delays.append(delay_days)
                    print(f"Task {task}: delay = {delay_days} days")  # Debug print
        self.kpi_results['average_delay_days'] = sum(delays) / len(delays) if delays else 0.0

        # 3. Procent godzin w terminie
        on_time_hours = 0
        total_hours = 0
        for task in tasks:
            task_rows = self.harmonogram_data[self.harmonogram_data['zadanie'] == task]
            deadline = pd.to_datetime(self.jobs_data[self.jobs_data['job_id'] == task]['deadline']).dt.date.iloc[0]
            for _, row in task_rows.iterrows():
                task_date = pd.to_datetime(row['data']).date()
                hours = row['czas']
                total_hours += hours
                if task_date <= deadline:
                    on_time_hours += hours
        self.kpi_results['on_time_hours_percentage'] = (on_time_hours / total_hours * 100) if total_hours > 0 else 0.0

        # 4. Wykorzystanie pracowników
        workers = self.harmonogram_data['pracownik'].unique()
        total_available_hours = len(self.working_days) * self.hours_per_day
        for worker in workers:
            worker_hours = self.harmonogram_data[self.harmonogram_data['pracownik'] == worker]['czas'].sum()
            utilization = (worker_hours / total_available_hours * 100) if total_available_hours > 0 else 0.0
            self.kpi_results['worker_utilization'][worker] = utilization

        # 5. Liczba zadań przypisanych do każdego pracownika
        for worker in workers:
            tasks_assigned = self.harmonogram_data[self.harmonogram_data['pracownik'] == worker]['zadanie'].unique()
            self.kpi_results['tasks_per_worker'][worker] = len(tasks_assigned)

        # 6. Średni czas realizacji zadania
        task_durations = []
        for task in tasks:
            task_rows = self.harmonogram_data[self.harmonogram_data['zadanie'] == task]
            start_date = pd.to_datetime(task_rows['data']).dt.date.min()
            end_date = pd.to_datetime(task_rows['data']).dt.date.max()
            # Count working days including start_date
            start_idx = self._get_day_index(start_date)
            end_idx = self._get_day_index(end_date)
            duration_days = max(1, end_idx - start_idx + 1)  # Ensure at least 1 day
            task_durations.append(duration_days)
            print(f"Task {task}: duration = {duration_days} days")  # Debug print
        self.kpi_results['average_task_duration_days'] = sum(task_durations) / len(task_durations) if len(task_durations) > 0 else 0.0

        return self.kpi_results

    def plot_delay_histogram(self):
        """Plot histogram of task delays in working days."""
        tasks = self.harmonogram_data['zadanie'].unique()
        delays = []
        for task in tasks:
            task_rows = self.harmonogram_data[self.harmonogram_data['zadanie'] == task]
            if any(task_rows['uwagi'] == 'Po terminie'):
                end_date = pd.to_datetime(task_rows['data']).dt.date.max()
                deadline = pd.to_datetime(self.jobs_data[self.jobs_data['job_id'] == task]['deadline']).dt.date.iloc[0]
                if end_date > deadline:
                    delay_days = self._count_all_days(deadline, end_date)
                    delays.append(delay_days)
        
        plt.figure(figsize=(10, 6))
        if delays:
            plt.hist(delays, bins=range(max(delays) + 2), align='left', rwidth=0.8)
        else:
            plt.hist([0], bins=1, rwidth=0.8)
        plt.title("Histogram of Task Delays (All Days)")
        plt.xlabel("Delay (all days)")
        plt.ylabel("Number of Tasks")
        plt.grid(True, alpha=0.3)
        plt.savefig("delay_histogram.png")
        plt.close()
        print("Delay histogram saved to delay_histogram.png")

    def plot_worker_daily_utilization(self):
        """Plot bar chart of worker hours per day with separate bars per worker."""
        workers = self.harmonogram_data['pracownik'].unique()
        dates = [d.strftime('%Y-%m-%d') for d in self.working_days]
        hours_per_worker_day = {worker: [0] * len(dates) for worker in workers}

        for _, row in self.harmonogram_data.iterrows():
            date = pd.to_datetime(row['data']).date()
            if date in self.working_days:
                worker = row['pracownik']
                hours = row['czas']
                date_idx = self.working_days.index(date)
                hours_per_worker_day[worker][date_idx] += hours

        bar_width = 0.25
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(dates))

        for i, worker in enumerate(workers):
            ax.bar(x + i * bar_width, hours_per_worker_day[worker], bar_width, label=worker)

        ax.set_title("Worker Hours per Day")
        ax.set_xlabel("Date")
        ax.set_ylabel("Hours Worked")
        ax.set_xticks(x + bar_width * (len(workers) - 1) / 2)
        ax.set_xticklabels(dates, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("worker_daily_utilization.png")
        plt.close()
        print("Worker daily utilization plot saved to worker_daily_utilization.png")

    def display_kpi(self):
        """Display KPIs in a readable format."""
        if not self.kpi_results:
            print("No KPI data available. Call calculate_kpi() first.")
            return

        print("\n=== KPI REPORT ===")
        print(f"1. Percentage of tasks completed on time: {self.kpi_results['on_time_percentage']:.2f}%")
        print(f"2. Average delay for late tasks: {self.kpi_results['average_delay_days']:.2f} all days")
        print(f"3. Percentage of hours completed on time: {self.kpi_results['on_time_hours_percentage']:.2f}%")
        print("4. Worker utilization:")
        for worker, utilization in self.kpi_results['worker_utilization'].items():
            print(f"   {worker}: {utilization:.2f}%")
        print("5. Number of tasks per worker:")
        for worker, count in self.kpi_results['tasks_per_worker'].items():
            print(f"   {worker}: {count} tasks")
        print(f"6. Average task duration: {self.kpi_results['average_task_duration_days']:.2f} working days")

    def export_kpi_to_csv(self, filename: str = "kpi_report.csv"):
        """Export KPIs to a CSV file."""
        if not self.kpi_results:
            print("No KPI data to export. Call calculate_kpi() first.")
            return

        csv_data = [
            {"KPI": "Percentage of tasks completed on time", "Value": f"{self.kpi_results['on_time_percentage']:.2f}%"},
            {"KPI": "Average delay for late tasks", "Value": f"{self.kpi_results['average_delay_days']:.2f} all days"},
            {"KPI": "Percentage of hours completed on time", "Value": f"{self.kpi_results['on_time_hours_percentage']:.2f}%"},
        ]

        for worker, utilization in self.kpi_results['worker_utilization'].items():
            csv_data.append({"KPI": f"Utilization of {worker}", "Value": f"{utilization:.2f}%"})
        for worker, count in self.kpi_results['tasks_per_worker'].items():
            csv_data.append({"KPI": f"Tasks assigned to {worker}", "Value": f"{count}"})
        csv_data.append({"KPI": "Average task duration", "Value": f"{self.kpi_results['average_task_duration_days']:.2f} working days"})

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['KPI', 'Value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                for row in csv_data:
                    writer.writerow(row)
            print(f"KPI report exported successfully to {filename}")
        except Exception as e:
            print(f"Error exporting KPI to CSV: {e}")

    def example_usage(self):
        """Example usage of KPICalculator."""
        try:
            self.load_data()
            self.calculate_kpi()
            self.display_kpi()
            self.plot_delay_histogram()
            self.plot_worker_daily_utilization()
            self.export_kpi_to_csv()
        except Exception as e:
            print(f"Error in KPI calculation or visualization: {e}")

if __name__ == "__main__":
    calculator = KPICalculator("harmonogram.csv", "jobs.csv")
    calculator.example_usage()