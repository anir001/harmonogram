from ortools.sat.python import cp_model
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Tuple, Optional
import numpy as np
import csv
import math
from kpi import KPICalculator

class ProductionScheduler:
    def __init__(self):
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
        # Input data
        self.jobs = []
        self.workers = []
        self.working_days = []
        self.day_to_index = {}
        
        # Decision variables
        self.assigned = {}  # assigned[job_id][worker_id]
        self.start_day = {}  # start_day[job_id] - starting day
        self.end_day = {}   # end_day[job_id] - ending day
        self.job_active = {} # job_active[job_id][day_idx] - is job active on this day
        self.work_hours = {} # work_hours[job_id][worker_id][day_idx] - hours worked
        
        # Parameters
        self.hours_per_day = 8
        self.start_date = None
        self.end_date = None
        
    def import_jobs_from_csv(self, filename: str):
        """Import jobs from CSV file with format: job_id;duration;start_date;deadline;max_parallel"""
        try:
            with open(filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=';')
                required_fields = {'job_id', 'duration', 'start_date', 'deadline'}
                if not all(field in reader.fieldnames for field in required_fields):
                    raise ValueError("CSV file missing required fields: job_id, duration, start_date, deadline")
                
                for row in reader:
                    try:
                        job_id = row['job_id']
                        duration = int(row['duration'])
                        start_date = datetime.strptime(row['start_date'], '%Y-%m-%d')
                        deadline = datetime.strptime(row['deadline'], '%Y-%m-%d')
                        max_parallel = int(row.get('max_parallel', 1))
                        
                        self.add_job(job_id, duration, start_date, deadline, max_parallel)
                    except (ValueError, KeyError) as e:
                        print(f"Error processing job {row.get('job_id', 'unknown')}: {e}")
                        continue
            
            print(f"Successfully imported {len(self.jobs)} jobs from {filename}")
        except Exception as e:
            print(f"Error reading jobs CSV file: {e}")
    
    def import_workers_from_csv(self, filename: str):
        """Import workers from CSV file with format: worker_id;unavailable_start;unavailable_end"""
        try:
            with open(filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=';')
                if 'worker_id' not in reader.fieldnames:
                    raise ValueError("CSV file missing required field: worker_id")
                
                worker_unavailability = {}
                for row in reader:
                    worker_id = row['worker_id']
                    if worker_id not in worker_unavailability:
                        worker_unavailability[worker_id] = []
                    
                    if row.get('unavailable_start') and row.get('unavailable_end'):
                        try:
                            start_unavail = datetime.strptime(row['unavailable_start'], '%Y-%m-%d')
                            end_unavail = datetime.strptime(row['unavailable_end'], '%Y-%m-%d')
                            worker_unavailability[worker_id].append((start_unavail, end_unavail))
                        except ValueError as e:
                            print(f"Error processing unavailability for worker {worker_id}: {e}")
                            continue
                
                for worker_id, unavailable_ranges in worker_unavailability.items():
                    self.add_worker(worker_id, unavailable_ranges)
            
            print(f"Successfully imported {len(self.workers)} workers from {filename}")
        except Exception as e:
            print(f"Error reading workers CSV file: {e}")
    
    def add_job(self, job_id: str, duration: int, start_date: datetime, 
                deadline: datetime, max_parallel: int = 1):
        """Add a job to the model"""
        job = {
            'job_id': job_id,
            'duration': duration,
            'start_date': start_date,
            'deadline': deadline,
            'max_parallel': max_parallel
        }
        self.jobs.append(job)
        
    def add_worker(self, worker_id: str, unavailable_ranges: List[Tuple[datetime, datetime]] = None):
        """Add a worker to the model"""
        worker = {
            'worker_id': worker_id,
            'unavailable_ranges': unavailable_ranges or []
        }
        self.workers.append(worker)
        
    def _generate_working_days(self, start_date: datetime, end_date: datetime):
        """Generate list of working days (Monday-Friday) in the given period"""
        working_days = []
        current_date = start_date
        
        while current_date <= end_date:
            # 0 = Monday, 4 = Friday
            if current_date.weekday() < 5:
                working_days.append(current_date.date())
            current_date += timedelta(days=1)
            
        return working_days
    
    def _is_worker_available(self, worker_id: str, date) -> bool:
        """Check if worker is available on given date"""
        worker = next(w for w in self.workers if w['worker_id'] == worker_id)
        
        for start_unavail, end_unavail in worker['unavailable_ranges']:
            if start_unavail.date() <= date <= end_unavail.date():
                return False
        return True
    
    def _calculate_required_days(self, duration: int, max_parallel: int) -> int:
        # Oblicz minimalną liczbę dni potrzebną, uwzględniając max_parallel
        return math.ceil(duration / (self.hours_per_day * max_parallel))
    
    def build_model(self, planning_horizon_days: int = 30):
        """Build the CP-SAT model with continuous job execution"""
        if not self.jobs or not self.workers:
            raise ValueError("No jobs or workers defined")
            
        # Determine planning range
        min_start = min(job['start_date'] for job in self.jobs)
        max_deadline = max(job['deadline'] for job in self.jobs)
        
        self.start_date = min_start
        self.end_date = max(max_deadline, min_start + timedelta(days=planning_horizon_days))
        
        # Generate working days
        self.working_days = self._generate_working_days(self.start_date, self.end_date)
        self.day_to_index = {day: idx for idx, day in enumerate(self.working_days)}
        
        num_days = len(self.working_days)
        
        print(f"Planning period: {self.start_date.date()} - {self.end_date.date()}")
        print(f"Number of working days: {num_days}")
        
        # === DECISION VARIABLES ===
        
        # 1. Worker assignment to jobs
        for job in self.jobs:
            self.assigned[job['job_id']] = {}
            for worker in self.workers:
                self.assigned[job['job_id']][worker['worker_id']] = \
                    self.model.NewBoolVar(f"assigned_{job['job_id']}_{worker['worker_id']}")
        
        # 2. Job start and end days
        for job in self.jobs:
            job_start_idx = max(0, self._get_day_index(job['start_date'].date()))
            job_end_idx = num_days - 1  # Allow end day up to the planning horizon
            
            # Calculate minimum required days for this job
            required_days = self._calculate_required_days(job['duration'], job['max_parallel'])
            
            self.start_day[job['job_id']] = self.model.NewIntVar(
                job_start_idx, num_days - required_days, f"start_day_{job['job_id']}")
            
            self.end_day[job['job_id']] = self.model.NewIntVar(
                job_start_idx + required_days - 1, job_end_idx, f"end_day_{job['job_id']}")
                
        # 3. Job activity on each day (binary)
        for job in self.jobs:
            self.job_active[job['job_id']] = {}
            for day_idx in range(num_days):
                self.job_active[job['job_id']][day_idx] = \
                    self.model.NewBoolVar(f"job_active_{job['job_id']}_{day_idx}")
        
        # 4. Work hours per job/worker/day
        for job in self.jobs:
            self.work_hours[job['job_id']] = {}
            for worker in self.workers:
                self.work_hours[job['job_id']][worker['worker_id']] = {}
                for day_idx in range(num_days):
                    max_hours = min(job['duration'], self.hours_per_day)
                    self.work_hours[job['job_id']][worker['worker_id']][day_idx] = \
                        self.model.NewIntVar(0, max_hours, 
                                           f"work_hours_{job['job_id']}_{worker['worker_id']}_{day_idx}")
        
        # === CONSTRAINTS ===
        
        # 1. Each job must be assigned to at least one worker, up to max_parallel
        for job in self.jobs:
            self.model.Add(
                sum(self.assigned[job['job_id']][worker['worker_id']] 
                    for worker in self.workers) >= 1
            )
            self.model.Add(
                sum(self.assigned[job['job_id']][worker['worker_id']] 
                    for worker in self.workers) <= job['max_parallel']
            )
        
        # 2. Job duration constraints
        for job in self.jobs:
            for worker in self.workers:
                total_hours = sum(
                    self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                    for day_idx in range(num_days)
                )
                # If not assigned, cannot work on it
                self.model.Add(
                    total_hours == 0
                ).OnlyEnforceIf(self.assigned[job['job_id']][worker['worker_id']].Not())
                # Remove incorrect constraint limiting total hours to hours_per_day
                # If assigned, hours are constrained per day below
                for day_idx in range(num_days):
                    self.model.Add(
                        self.work_hours[job['job_id']][worker['worker_id']][day_idx] == 0
                    ).OnlyEnforceIf(self.assigned[job['job_id']][worker['worker_id']].Not())
                    self.model.Add(
                        self.work_hours[job['job_id']][worker['worker_id']][day_idx] <= self.hours_per_day
                    ).OnlyEnforceIf(self.assigned[job['job_id']][worker['worker_id']])
                    
        # 3. Job activity linking
        for job in self.jobs:
            for day_idx in range(num_days):
                # Job is active on a day if any worker works on it that day
                total_work_today = sum(
                    self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                    for worker in self.workers
                )
                # Link job_active with actual work
                self.model.Add(total_work_today >= 1).OnlyEnforceIf(
                    self.job_active[job['job_id']][day_idx])
                self.model.Add(total_work_today == 0).OnlyEnforceIf(
                    self.job_active[job['job_id']][day_idx].Not())
                # Ensure job is only active after start_date
                if self.working_days[day_idx] < job['start_date'].date():
                    self.model.Add(self.job_active[job['job_id']][day_idx] == 0)
        
        # 4. ENHANCED CONTINUITY CONSTRAINTS - Ensure continuous execution
        for job in self.jobs:
            required_days = self._calculate_required_days(job['duration'], job['max_parallel'])
            
            # Ensure sufficient days to cover duration
            active_days = [self.job_active[job['job_id']][day_idx] for day_idx in range(num_days)]
            self.model.Add(sum(active_days) >= math.ceil(job['duration'] / (self.hours_per_day * job['max_parallel'])))
                        
            for day_idx in range(num_days):
                # If day_idx is before start_day, job cannot be active
                is_before_start = self.model.NewBoolVar(f"before_start_{job['job_id']}_{day_idx}")
                self.model.Add(day_idx < self.start_day[job['job_id']]).OnlyEnforceIf(is_before_start)
                self.model.Add(day_idx >= self.start_day[job['job_id']]).OnlyEnforceIf(is_before_start.Not())
                self.model.Add(self.job_active[job['job_id']][day_idx] == 0).OnlyEnforceIf(is_before_start)
                
                # If day_idx is after end_day, job cannot be active
                is_after_end = self.model.NewBoolVar(f"after_end_{job['job_id']}_{day_idx}")
                self.model.Add(day_idx > self.end_day[job['job_id']]).OnlyEnforceIf(is_after_end)
                self.model.Add(day_idx <= self.end_day[job['job_id']]).OnlyEnforceIf(is_after_end.Not())
                self.model.Add(self.job_active[job['job_id']][day_idx] == 0).OnlyEnforceIf(is_after_end)
                
                # Ensure hours are assigned only in active days
                for worker in self.workers:
                    hours_var = self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                    # If not active, no hours can be assigned
                    self.model.Add(hours_var == 0).OnlyEnforceIf(self.job_active[job['job_id']][day_idx].Not())
                    # If active and assigned to worker, hours can be up to hours_per_day
                    self.model.Add(hours_var <= self.hours_per_day).OnlyEnforceIf([
                        self.job_active[job['job_id']][day_idx],
                        self.assigned[job['job_id']][worker['worker_id']]
                    ])
            
            # Ensure end_day is at least start_day and respects required_days
            self.model.Add(self.end_day[job['job_id']] >= self.start_day[job['job_id']])
            self.model.Add(self.end_day[job['job_id']] <= self.start_day[job['job_id']] + required_days - 1)
            
            # Allow hours to be assigned in active days for assigned workers
            for worker in self.workers:
                assigned = self.assigned[job['job_id']][worker['worker_id']]
                for day_idx in range(num_days):
                    hours_var = self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                    # Hours can only be assigned if job is active and worker is assigned
                    self.model.Add(hours_var <= self.hours_per_day).OnlyEnforceIf([
                        self.job_active[job['job_id']][day_idx],
                        assigned
                    ])
                    self.model.Add(hours_var == 0).OnlyEnforceIf(assigned.Not())
            
            # Ensure total hours across all workers equals job duration
            total_hours_assigned = sum(
                self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                for worker in self.workers
                for day_idx in range(num_days)
            )
            self.model.Add(total_hours_assigned == job['duration'])
            
            # Encourage max_parallel workers to be assigned when possible
            workers_assigned_count = sum(
            self.assigned[job['job_id']][worker['worker_id']]
            for worker in self.workers
            )
            self.model.Add(workers_assigned_count == job['max_parallel'])
            
            # Ensure hours per worker are consistent with assignment
            for worker in self.workers:
                hours_vars = [self.work_hours[job['job_id']][worker['worker_id']][day_idx] for day_idx in range(num_days)]
                total_worker_hours = sum(hours_vars)
                # If assigned, worker must contribute some hours
                self.model.Add(total_worker_hours > 0).OnlyEnforceIf(self.assigned[job['job_id']][worker['worker_id']])
                self.model.Add(total_worker_hours == 0).OnlyEnforceIf(self.assigned[job['job_id']][worker['worker_id']].Not())
                # Allow flexible hour distribution in active days
                for day_idx in range(num_days):
                    hours_var = self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                    self.model.Add(hours_var <= self.hours_per_day).OnlyEnforceIf([
                        self.job_active[job['job_id']][day_idx],
                        self.assigned[job['job_id']][worker['worker_id']]
                    ])
        
        # 5. Worker daily capacity constraints
        for worker in self.workers:
            for day_idx in range(num_days):
                daily_hours = sum(
                    self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                    for job in self.jobs
                )
                self.model.Add(daily_hours <= self.hours_per_day)
        
        # 6. Worker availability constraints
        for worker in self.workers:
            for day_idx, day in enumerate(self.working_days):
                if not self._is_worker_available(worker['worker_id'], day):
                    # Worker cannot work on unavailable days
                    for job in self.jobs:
                        self.model.Add(
                            self.work_hours[job['job_id']][worker['worker_id']][day_idx] == 0
                        )
        # 7. Max parallel workers per job constraint
        for job in self.jobs:
            for day_idx in range(num_days):
                # Count number of workers assigned to this job on this day
                workers_assigned = [
                    self.model.NewBoolVar(f"worker_assigned_{job['job_id']}_{worker['worker_id']}_{day_idx}")
                    for worker in self.workers
                ]
                for idx, worker in enumerate(self.workers):
                    hours_var = self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                    self.model.Add(hours_var > 0).OnlyEnforceIf(workers_assigned[idx])
                    self.model.Add(hours_var == 0).OnlyEnforceIf(workers_assigned[idx].Not())
                self.model.Add(sum(workers_assigned) <= job['max_parallel'])
        # # 8. Sequential execution for ALL jobs
        # for worker in self.workers:
        #     # For each pair of jobs that could be assigned to this worker
        #     for i, job1 in enumerate(self.jobs):
        #         for j, job2 in enumerate(self.jobs):
        #             if i >= j:  # Only consider unique pairs and avoid self-comparison
        #                 continue
                    
        #             # Both jobs assigned to same worker
        #             both_assigned = self.model.NewBoolVar(
        #                 f"both_assigned_{worker['worker_id']}_{job1['job_id']}_{job2['job_id']}")
                    
        #             self.model.AddBoolAnd([
        #                 self.assigned[job1['job_id']][worker['worker_id']],
        #                 self.assigned[job2['job_id']][worker['worker_id']]
        #             ]).OnlyEnforceIf(both_assigned)
                    
        #             # If both jobs assigned to same worker, they cannot overlap
        #             job1_before_job2 = self.model.NewBoolVar(
        #                 f"job1_before_job2_{worker['worker_id']}_{job1['job_id']}_{job2['job_id']}")
                    
        #             # Job1 completely before Job2: end_day[job1] < start_day[job2]
        #             self.model.Add(
        #                 self.end_day[job1['job_id']] < self.start_day[job2['job_id']]
        #             ).OnlyEnforceIf([both_assigned, job1_before_job2])
                    
        #             # Job2 completely before Job1: end_day[job2] < start_day[job1]
        #             self.model.Add(
        #                 self.end_day[job2['job_id']] < self.start_day[job1['job_id']]
        #             ).OnlyEnforceIf([both_assigned, job1_before_job2.Not()])
        
        # 9. Encourage completing jobs on same day when possible
        for job in self.jobs:
            if job['duration'] <= self.hours_per_day:
                # For jobs that can fit in one day, encourage start_day == end_day
                same_day = self.model.NewBoolVar(f"same_day_{job['job_id']}")
                self.model.Add(self.start_day[job['job_id']] == self.end_day[job['job_id']]).OnlyEnforceIf(same_day)
                
                # If completed on same day, all work should be on that day
                for worker in self.workers:
                    for day_idx in range(num_days):
                        is_the_day = self.model.NewBoolVar(f"is_the_day_{job['job_id']}_{worker['worker_id']}_{day_idx}")
                        
                        self.model.AddBoolAnd([
                            same_day,
                            self.assigned[job['job_id']][worker['worker_id']]
                        ]).OnlyEnforceIf(is_the_day)
                        
                        # If it's the day and job is assigned to this worker, work full duration
                        is_start_day = self.model.NewBoolVar(f"is_start_day_{job['job_id']}_{day_idx}")
                        self.model.Add(self.start_day[job['job_id']] == day_idx).OnlyEnforceIf(is_start_day)
                        self.model.Add(self.start_day[job['job_id']] != day_idx).OnlyEnforceIf(is_start_day.Not())
                        
                        self.model.AddBoolAnd([
                            is_the_day,
                            is_start_day
                        ]).OnlyEnforceIf(self.model.NewBoolVar(f"work_full_{job['job_id']}_{worker['worker_id']}_{day_idx}"))
                        
                        # Work all duration on the single day
                        self.model.Add(
                            self.work_hours[job['job_id']][worker['worker_id']][day_idx] == job['duration']
                        ).OnlyEnforceIf([same_day, self.assigned[job['job_id']][worker['worker_id']], is_start_day])
    
    def _get_day_index(self, date) -> int:
        """Return the index of the day in working_days list"""
        if date in self.day_to_index:
            return self.day_to_index[date]
        # Find the nearest working day
        for i, working_day in enumerate(self.working_days):
            if working_day >= date:
                return i
        return len(self.working_days) - 1
    
    # def set_objective(self, objective_type: str = "makespan"):
        """Set the objective function"""
        if objective_type == "makespan":
            # Minimize maximum completion time
            makespan = self.model.NewIntVar(0, len(self.working_days), "makespan")
            for job in self.jobs:
                self.model.Add(makespan >= self.end_day[job['job_id']])
            self.model.Minimize(makespan)
            
        elif objective_type == "workload_balance":
            # Balance worker workload
            max_workload = self.model.NewIntVar(0, len(self.working_days) * self.hours_per_day, "max_workload")
            for worker in self.workers:
                total_hours = sum(
                    self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                    for job in self.jobs
                    for day_idx in range(len(self.working_days))
                )
                self.model.Add(max_workload >= total_hours)
            self.model.Minimize(max_workload)
            
        elif objective_type == "minimize_delays":
            # Minimize delays relative to deadlines - POPRAWIONA WERSJA
            delay_vars = []
            
            for job in self.jobs:
                # Znajdź indeks dnia deadline
                deadline_date = job['deadline'].date()
                deadline_idx = self._get_day_index(deadline_date)
                
                # Upewnij się, że deadline_idx jest w prawidłowym zakresie
                deadline_idx = min(deadline_idx, len(self.working_days) - 1)
                deadline_idx = max(deadline_idx, 0)
                
                # Utwórz zmienną opóźnienia dla tego zadania
                max_possible_delay = len(self.working_days) - deadline_idx
                delay_var = self.model.NewIntVar(0, max_possible_delay, f"delay_{job['job_id']}")
                
                # Oblicz opóźnienie: max(0, end_day - deadline_idx)
                # Używamy pomocniczej zmiennej do reprezentacji różnicy
                diff_var = self.model.NewIntVar(-len(self.working_days), len(self.working_days), 
                                            f"diff_{job['job_id']}")
                self.model.Add(diff_var == self.end_day[job['job_id']] - deadline_idx)
                
                # delay = max(0, diff)
                self.model.AddMaxEquality(delay_var, [diff_var, 0])
                
                delay_vars.append(delay_var)
                
                print(f"Job {job['job_id']}: deadline={deadline_date}, deadline_idx={deadline_idx}, "
                    f"max_delay={max_possible_delay}")
            
            # Minimalizuj sumę wszystkich opóźnień
            total_delay = sum(delay_vars)
            self.model.Minimize(total_delay)
            
        elif objective_type == "pack_jobs":
            total_completion_time = sum(self.end_day[job['job_id']] for job in self.jobs)
            weighted_hours = sum(
                day_idx * self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                for job in self.jobs
                for worker in self.workers
                for day_idx in range(len(self.working_days))
            )
            # Penalize under-utilization of hours
            total_hours_assigned = sum(
                self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                for job in self.jobs
                for worker in self.workers
                for day_idx in range(len(self.working_days))
            )
            hours_shortfall = sum(
                job['duration'] - sum(
                    self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                    for worker in self.workers
                    for day_idx in range(len(self.working_days))
                )
                for job in self.jobs
            )
            # Penalize delays past deadline
            total_delay = sum(
                self.model.NewIntVar(0, len(self.working_days), f"delay_{job['job_id']}") 
                for job in self.jobs
            )
            for job in self.jobs:
                deadline_idx = self._get_day_index(job['deadline'].date())
                self.model.AddMaxEquality(
                    self.model.NewIntVar(0, len(self.working_days), f"delay_{job['job_id']}"),
                    [self.end_day[job['job_id']] - deadline_idx, 0]
                )
            self.model.Minimize(total_completion_time * 1000 + weighted_hours + hours_shortfall * 10000 + total_delay * 5000)
    
    def set_objective(self, objective_type: str = "makespan"):
        """Set the objective function"""
        if objective_type == "makespan":
            # Minimize maximum completion time
            makespan = self.model.NewIntVar(0, len(self.working_days), "makespan")
            for job in self.jobs:
                self.model.Add(makespan >= self.end_day[job['job_id']])
            self.model.Minimize(makespan)
            
        elif objective_type == "workload_balance":
            # Balance worker workload
            max_workload = self.model.NewIntVar(0, len(self.working_days) * self.hours_per_day, "max_workload")
            for worker in self.workers:
                total_hours = sum(
                    self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                    for job in self.jobs
                    for day_idx in range(len(self.working_days))
                )
                self.model.Add(max_workload >= total_hours)
            self.model.Minimize(max_workload)
            
        elif objective_type == "minimize_delays":
            # Minimize delays relative to deadlines - POPRAWIONA WERSJA
            delay_vars = []
            
            for job in self.jobs:
                # Znajdź indeks dnia deadline
                deadline_date = job['deadline'].date()
                deadline_idx = self._get_day_index(deadline_date)
                
                # Upewnij się, że deadline_idx jest w prawidłowym zakresie
                deadline_idx = min(deadline_idx, len(self.working_days) - 1)
                deadline_idx = max(deadline_idx, 0)
                
                # Utwórz zmienną opóźnienia dla tego zadania
                max_possible_delay = len(self.working_days) - deadline_idx
                delay_var = self.model.NewIntVar(0, max_possible_delay, f"delay_{job['job_id']}")
                
                # Oblicz opóźnienie: max(0, end_day - deadline_idx)
                # Używamy pomocniczej zmiennej do reprezentacji różnicy
                diff_var = self.model.NewIntVar(-len(self.working_days), len(self.working_days), 
                                            f"diff_{job['job_id']}")
                self.model.Add(diff_var == self.end_day[job['job_id']] - deadline_idx)
                
                # delay = max(0, diff)
                self.model.AddMaxEquality(delay_var, [diff_var, 0])
                
                delay_vars.append(delay_var)
                
                print(f"Job {job['job_id']}: deadline={deadline_date}, deadline_idx={deadline_idx}, "
                    f"max_delay={max_possible_delay}")
            
            # Minimalizuj sumę wszystkich opóźnień
            total_delay = sum(delay_vars)
            self.model.Minimize(total_delay)
            
            print(f"minimize_delays: Created {len(delay_vars)} delay variables")
            
        elif objective_type == "pack_jobs":
            # Pack jobs efficiently - RÓWNIEŻ POPRAWIONA
            total_completion_time = sum(self.end_day[job['job_id']] for job in self.jobs)
            
            # Weighted completion time (earlier completion preferred)
            weighted_hours = sum(
                day_idx * self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                for job in self.jobs
                for worker in self.workers
                for day_idx in range(len(self.working_days))
            )
            
            # Penalize delays past deadline
            delay_penalty = 0
            for job in self.jobs:
                deadline_date = job['deadline'].date()
                deadline_idx = self._get_day_index(deadline_date)
                deadline_idx = min(deadline_idx, len(self.working_days) - 1)
                deadline_idx = max(deadline_idx, 0)
                
                # Utwórz zmienną opóźnienia
                max_possible_delay = len(self.working_days) - deadline_idx
                delay_var = self.model.NewIntVar(0, max_possible_delay, f"pack_delay_{job['job_id']}")
                
                # Oblicz opóźnienie
                diff_var = self.model.NewIntVar(-len(self.working_days), len(self.working_days), 
                                            f"pack_diff_{job['job_id']}")
                self.model.Add(diff_var == self.end_day[job['job_id']] - deadline_idx)
                self.model.AddMaxEquality(delay_var, [diff_var, 0])
                
                delay_penalty += delay_var * 5000  # Wysokie kary za opóźnienia
            
            # Kombinowana funkcja celu
            self.model.Minimize(
                total_completion_time * 1000 +  # Preferuj wcześniejsze zakończenie
                weighted_hours +                # Preferuj wcześniejsze wykonanie pracy  
                delay_penalty                   # Karaj opóźnienia
            )
        
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")
    
    def solve(self, time_limit_seconds: int = 300, debug=False) -> Dict:
        """Solve the model and return results"""
        result = {
            'status': None,
            'objective_value': None,
            'assignments': {},
            'schedule': {},
            'statistics': {
                'solve_time': 0.0,
                'num_variables': self.model.Proto().variables.__len__(),
                'num_constraints': self.model.Proto().constraints.__len__()
            }
        }
        
        self.solver.parameters.max_time_in_seconds = time_limit_seconds
        self.solver.parameters.log_search_progress = True
        self.solver.parameters.log_to_stdout = True
        status = self.solver.Solve(self.model)
        result['status'] = self.solver.StatusName(status)
        result['statistics']['solve_time'] = self.solver.WallTime()
        
        print(f"Solver status: {result['status']}")
        if status == cp_model.INFEASIBLE:
            print("Model is INFEASIBLE. Check constraints for conflicts.")
            for job in self.jobs:
                print(f"Job {job['job_id']}: duration={job['duration']}, start_date={job['start_date'].date()}, "
                    f"deadline={job['deadline'].date()}, max_parallel={job['max_parallel']}")
                required_days = self._calculate_required_days(job['duration'], job['max_parallel'])
                print(f"  Required days: {required_days}")
                print(f"  Available working days after start_date: {[d for d in self.working_days if d >= job['start_date'].date()]}")
            return result
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            result['objective_value'] = self.solver.ObjectiveValue()
            print("Assigned workers:")
            for job in self.jobs:
                result['assignments'][job['job_id']] = []
                result['schedule'][job['job_id']] = {
                    'workers': [],
                    'start_date': None,
                    'end_date': None,
                    'duration_hours': job['duration'],
                    'work_details': {}
                }
                start_idx = self.solver.Value(self.start_day[job['job_id']])
                end_idx = self.solver.Value(self.end_day[job['job_id']])
                result['schedule'][job['job_id']]['start_date'] = self.working_days[start_idx]
                result['schedule'][job['job_id']]['end_date'] = self.working_days[end_idx]
                
                total_hours = 0
                for worker in self.workers:
                    if self.solver.Value(self.assigned[job['job_id']][worker['worker_id']]):
                        print(f"Job {job['job_id']} assigned to {worker['worker_id']}")
                        result['assignments'][job['job_id']].append(worker['worker_id'])
                        result['schedule'][job['job_id']]['workers'].append(worker['worker_id'])
                        
                        for day_idx in range(len(self.working_days)):
                            hours = self.solver.Value(
                                self.work_hours[job['job_id']][worker['worker_id']][day_idx]
                            )
                            if hours > 0:
                                result['schedule'][job['job_id']]['work_details'][
                                    (self.working_days[day_idx], worker['worker_id'])
                                ] = hours
                                total_hours += hours
                
                deadline_idx = self._get_day_index(job['deadline'].date())
                print(f"Job {job['job_id']}: Total hours: {total_hours}h (expected: {job['duration']}h), "
                    f"End day: {self.working_days[end_idx]} (deadline: {job['deadline'].date()})")
        
        return result
    
    def export_to_csv(self, result: Dict, filename: str = "harmonogram.csv"):
        """Export schedule to CSV file with format: data;zadanie;czas;pracownik;uwagi"""
        if not result['schedule']:
            print("No schedule data to export")
            return
        
        # Prepare data for CSV
        csv_data = []
        
        for job_id, schedule in result['schedule'].items():
            work_details = schedule['work_details']
            job = next(j for j in self.jobs if j['job_id'] == job_id)
            is_late = schedule['end_date'] > job['deadline'].date()
            uwagi = "Po terminie" if is_late else ""
            
            for (date, worker), hours in work_details.items():
                csv_data.append({
                    'data': date.strftime('%Y-%m-%d'),
                    'zadanie': job_id,
                    'czas': hours,
                    'pracownik': worker,
                    'uwagi': uwagi
                })
        
        # Sort by date, then by worker, then by task
        csv_data.sort(key=lambda x: (x['data'], x['pracownik'], x['zadanie']))
        
        # Write to CSV file
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['data', 'zadanie', 'czas', 'pracownik', 'uwagi']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                for row in csv_data:
                    writer.writerow(row)
            
            print(f"Schedule exported successfully to {filename}")
            print(f"Total records: {len(csv_data)}")
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
    
    def print_solution(self, result: Dict, debug=False):
        """Display the solution in a readable format"""
        print(f"\n=== SCHEDULING RESULTS ===")
        print(f"Status: {result['status']}")
        print(f"Objective value: {result['objective_value']}")
        print(f"Solve time: {result['statistics']['solve_time']:.2f}s")
        print(f"Variables: {result['statistics']['num_variables']}")
        print(f"Constraints: {result['statistics']['num_constraints']}")
        
        if debug:
            if result['schedule']:
                print(f"\n=== SCHEDULE ===")
                sorted_jobs = sorted(result['schedule'].items(), key=lambda x: x[1]['start_date'])
                
                for job_id, schedule in sorted_jobs:
                    job = next(j for j in self.jobs if j['job_id'] == job_id)
                    print(f"Job {job_id}:")
                    print(f"  Workers: {', '.join(schedule['workers'])}")
                    print(f"  Period: {schedule['start_date']} - {schedule['end_date']}")
                    print(f"  Duration: {schedule['duration_hours']}h")
                    print(f"  Deadline: {job['deadline'].date()}")
                    print(f"  Work schedule:")
                    
                    total_scheduled = 0
                    work_days = sorted(schedule['work_details'].items(), key=lambda x: (x[0][0], x[0][1]))
                    for i, ((date, worker), hours) in enumerate(work_days):
                        print(f"    {date} ({worker}): {hours}h")
                        total_scheduled += hours
                    
                    print(f"  Total scheduled: {total_scheduled}h")
                    
                    deadline_date = job['deadline'].date()
                    if schedule['end_date'] <= deadline_date:
                        print(f"  Status: ✓ On time")
                    else:
                        print(f"  Status: ⚠ Po terminie")
                    print()
                
                print(f"=== DAILY SCHEDULE SUMMARY ===")
                daily_schedule = {}
                for job_id, schedule in result['schedule'].items():
                    for (date, worker), hours in schedule['work_details'].items():
                        if date not in daily_schedule:
                            daily_schedule[date] = {}
                        if worker not in daily_schedule[date]:
                            daily_schedule[date][worker] = []
                        daily_schedule[date][worker].append(f"{job_id}({hours}h)")
                
                for date in sorted(daily_schedule.keys()):
                    print(f"{date}:")
                    for worker, jobs in daily_schedule[date].items():
                        total_hours = sum(int(job.split('(')[1].split('h')[0]) for job in jobs)
                        jobs_str = ', '.join(jobs)
                        print(f"  {worker}: {jobs_str} [Total: {total_hours}h]")
                    print()

# === EXAMPLE USAGE ===
def example_usage():
    """Example usage with continuous job execution"""
    scheduler = ProductionScheduler()
    
    # Import jobs and workers from CSV files
    scheduler.import_jobs_from_csv("jobs.csv")
    scheduler.import_workers_from_csv("workers.csv")
    
    # Build model
    scheduler.build_model()
    
    # Set optimization objective to pack jobs efficiently
    scheduler.set_objective("pack_jobs")
    
    # Solve
    result = scheduler.solve(time_limit_seconds=120)
    
    # Display results
    scheduler.print_solution(result)
    
    # Export to CSV
    scheduler.export_to_csv(result, "harmonogram.csv")
    
    return scheduler, result

def compare():
    """Example usage comparing all optimization methods"""
    import time
    from datetime import datetime
    
    # Initialize scheduler
    scheduler = ProductionScheduler()
    
    try:
        # Import data
        print("Loading data from CSV files...")
        scheduler.import_jobs_from_csv("jobs.csv")
        scheduler.import_workers_from_csv("workers.csv")
        
        if not scheduler.jobs or not scheduler.workers:
            print("Error: No jobs or workers loaded. Please check your CSV files.")
            return None, None
        
        print(f"Successfully loaded {len(scheduler.jobs)} jobs and {len(scheduler.workers)} workers")
        
        # Available optimization methods
        optimization_methods = [
            ("makespan", "Minimize Maximum Completion Time"),
            ("workload_balance", "Balance Worker Workload"),
            ("minimize_delays", "Minimize Delays Past Deadlines"),
            ("pack_jobs", "Pack Jobs Efficiently")
        ]
        
        # Store results for comparison
        comparison_results = {}
        
        print("\n" + "="*80)
        print("RUNNING OPTIMIZATION COMPARISON")
        print("="*80)
        
        for method_name, method_description in optimization_methods:
            print(f"\n🔄 Running optimization method: {method_name.upper()}")
            print(f"Description: {method_description}")
            print("-" * 50)
            
            # Create a fresh scheduler instance for each method
            method_scheduler = ProductionScheduler()
            method_scheduler.import_jobs_from_csv("jobs.csv")
            method_scheduler.import_workers_from_csv("workers.csv")
            
            # Build model
            method_scheduler.build_model()
            
            # Set objective
            method_scheduler.set_objective(method_name)
            
            # Solve with time limit
            start_time = time.time()
            result = method_scheduler.solve(time_limit_seconds=180, debug=False)
            solve_time = time.time() - start_time
            
            # Store results
            comparison_results[method_name] = {
                'result': result,
                'scheduler': method_scheduler,
                'method_description': method_description,
                'solve_time': solve_time
            }
            
            # Print basic results
            print(f"Status: {result['status']}")
            print(f"Objective Value: {result.get('objective_value', 'N/A')}")
            print(f"Solve Time: {solve_time:.2f}s")
            
            if result['status'] in ['OPTIMAL', 'FEASIBLE']:
                print("✅ Solution found successfully")
                # Export individual results
                filename = f"harmonogram_{method_name}.csv"
                method_scheduler.export_to_csv(result, filename)
                print(f"📁 Results exported to {filename}")
            else:
                print("❌ No feasible solution found")
        
        # Generate comprehensive comparison report
        print("\n" + "="*80)
        print("COMPREHENSIVE COMPARISON REPORT")
        print("="*80)
        
        generate_comparison_report(comparison_results)
        
        # Export comparison summary to CSV
        export_comparison_to_csv(comparison_results)
        
        return comparison_results, scheduler
        
    except Exception as e:
        print(f"Error during optimization comparison: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_comparison_report(comparison_results):
    """Generate detailed comparison report"""
    
    print("\n📊 PERFORMANCE SUMMARY")
    print("-" * 60)
    print(f"{'Method':<20} {'Status':<12} {'Obj Value':<12} {'Solve Time':<12}")
    print("-" * 60)
    
    successful_methods = []
    
    for method_name, data in comparison_results.items():
        result = data['result']
        status = result['status']
        obj_value = result.get('objective_value', 'N/A')
        solve_time = data['solve_time']
        
        print(f"{method_name:<20} {status:<12} {obj_value:<12} {solve_time:.2f}s")
        
        if status in ['OPTIMAL', 'FEASIBLE']:
            successful_methods.append(method_name)
    
    if not successful_methods:
        print("\n❌ No successful solutions found for comparison")
        return
    
    print(f"\n✅ Successful methods: {', '.join(successful_methods)}")
    
    # Detailed analysis for successful methods
    print("\n📋 DETAILED ANALYSIS")
    print("=" * 80)
    
    for method_name in successful_methods:
        data = comparison_results[method_name]
        result = data['result']
        scheduler = data['scheduler']
        
        print(f"\n🎯 METHOD: {method_name.upper()}")
        print(f"Description: {data['method_description']}")
        print("-" * 40)
        
        # Calculate key metrics
        metrics = calculate_schedule_metrics(result, scheduler)
        
        print(f"📈 Key Metrics:")
        print(f"  • Total jobs scheduled: {metrics['total_jobs']}")
        print(f"  • Jobs completed on time: {metrics['on_time_jobs']}")
        print(f"  • Jobs completed late: {metrics['late_jobs']}")
        print(f"  • On-time percentage: {metrics['on_time_percentage']:.1f}%")
        print(f"  • Average delay (late jobs): {metrics['avg_delay']:.1f} days")
        print(f"  • Total scheduled hours: {metrics['total_hours']}")
        print(f"  • Schedule span: {metrics['schedule_span']} days")
        print(f"  • Worker utilization:")
        
        for worker, util in metrics['worker_utilization'].items():
            print(f"    - {worker}: {util:.1f}%")
        
        # Print schedule summary
        print(f"\n📅 Schedule Summary:")
        for job_id, schedule in result['schedule'].items():
            job = next(j for j in scheduler.jobs if j['job_id'] == job_id)
            workers_str = ', '.join(schedule['workers'])
            status_emoji = "✅" if schedule['end_date'] <= job['deadline'].date() else "⚠️"
            
            print(f"  {status_emoji} Job {job_id}: {schedule['start_date']} → {schedule['end_date']} "
                  f"({workers_str}) [{schedule['duration_hours']}h]")

def calculate_schedule_metrics(result, scheduler):
    """Calculate comprehensive metrics for a schedule"""
    metrics = {
        'total_jobs': len(result['schedule']),
        'on_time_jobs': 0,
        'late_jobs': 0,
        'total_hours': 0,
        'total_delays': [],
        'worker_utilization': {},
        'schedule_span': 0
    }
    
    # Calculate job metrics
    earliest_start = None
    latest_end = None
    
    for job_id, schedule in result['schedule'].items():
        job = next(j for j in scheduler.jobs if j['job_id'] == job_id)
        
        if schedule['end_date'] <= job['deadline'].date():
            metrics['on_time_jobs'] += 1
        else:
            metrics['late_jobs'] += 1
            delay_days = (schedule['end_date'] - job['deadline'].date()).days
            metrics['total_delays'].append(delay_days)
        
        metrics['total_hours'] += schedule['duration_hours']
        
        if earliest_start is None or schedule['start_date'] < earliest_start:
            earliest_start = schedule['start_date']
        if latest_end is None or schedule['end_date'] > latest_end:
            latest_end = schedule['end_date']
    
    # Calculate derived metrics
    metrics['on_time_percentage'] = (metrics['on_time_jobs'] / metrics['total_jobs'] * 100) if metrics['total_jobs'] > 0 else 0
    metrics['avg_delay'] = sum(metrics['total_delays']) / len(metrics['total_delays']) if metrics['total_delays'] else 0
    
    if earliest_start and latest_end:
        metrics['schedule_span'] = (latest_end - earliest_start).days + 1
    
    # Calculate worker utilization
    total_available_hours = len(scheduler.working_days) * scheduler.hours_per_day
    
    for worker in scheduler.workers:
        worker_hours = 0
        for job_id, schedule in result['schedule'].items():
            for (date, worker_id), hours in schedule['work_details'].items():
                if worker_id == worker['worker_id']:
                    worker_hours += hours
        
        utilization = (worker_hours / total_available_hours * 100) if total_available_hours > 0 else 0
        metrics['worker_utilization'][worker['worker_id']] = utilization
    
    return metrics

def export_comparison_to_csv(comparison_results, filename="optimization_comparison.csv"):
    """Export comparison results to CSV"""
    import csv
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'Method', 'Description', 'Status', 'Objective_Value', 'Solve_Time_s',
                'Total_Jobs', 'On_Time_Jobs', 'Late_Jobs', 'On_Time_Percentage',
                'Avg_Delay_Days', 'Total_Hours', 'Schedule_Span_Days'
            ]
            
            # Add worker utilization columns
            if comparison_results:
                first_result = next(iter(comparison_results.values()))
                if first_result['result']['status'] in ['OPTIMAL', 'FEASIBLE']:
                    workers = [w['worker_id'] for w in first_result['scheduler'].workers]
                    for worker in workers:
                        fieldnames.append(f'Utilization_{worker}_%')
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            
            for method_name, data in comparison_results.items():
                result = data['result']
                
                row = {
                    'Method': method_name,
                    'Description': data['method_description'],
                    'Status': result['status'],
                    'Objective_Value': result.get('objective_value', ''),
                    'Solve_Time_s': f"{data['solve_time']:.2f}"
                }
                
                if result['status'] in ['OPTIMAL', 'FEASIBLE']:
                    metrics = calculate_schedule_metrics(result, data['scheduler'])
                    
                    row.update({
                        'Total_Jobs': metrics['total_jobs'],
                        'On_Time_Jobs': metrics['on_time_jobs'],
                        'Late_Jobs': metrics['late_jobs'],
                        'On_Time_Percentage': f"{metrics['on_time_percentage']:.1f}",
                        'Avg_Delay_Days': f"{metrics['avg_delay']:.1f}",
                        'Total_Hours': metrics['total_hours'],
                        'Schedule_Span_Days': metrics['schedule_span']
                    })
                    
                    # Add worker utilization
                    for worker, util in metrics['worker_utilization'].items():
                        row[f'Utilization_{worker}_%'] = f"{util:.1f}"
                
                writer.writerow(row)
        
        print(f"\n📊 Comparison summary exported to {filename}")
        
    except Exception as e:
        print(f"Error exporting comparison to CSV: {e}")

def run_kpi_analysis_on_best_solution(comparison_results):
    """Run KPI analysis on the best solution found"""
    
    # Find the best solution (prioritize on-time completion, then minimize delays)
    best_method = None
    best_score = -1
    
    for method_name, data in comparison_results.items():
        result = data['result']
        if result['status'] not in ['OPTIMAL', 'FEASIBLE']:
            continue
        
        metrics = calculate_schedule_metrics(result, data['scheduler'])
        # Scoring: prioritize on-time percentage, penalize delays
        score = metrics['on_time_percentage'] - (metrics['avg_delay'] * 10)
        
        if score > best_score:
            best_score = score
            best_method = method_name
    
    if best_method:
        print(f"\n🏆 BEST SOLUTION: {best_method.upper()}")
        print(f"Running KPI analysis on best solution...")
        
        # Run KPI analysis
        try:
            kpi_calculator = KPICalculator(f"harmonogram_{best_method}.csv", "jobs.csv")
            kpi_calculator.load_data()
            kpi_results = kpi_calculator.calculate_kpi()
            
            print("\n📈 KPI ANALYSIS OF BEST SOLUTION:")
            kpi_calculator.display_kpi()
            
            # Generate visualizations
            kpi_calculator.plot_delay_histogram()
            kpi_calculator.plot_worker_daily_utilization()
            kpi_calculator.export_kpi_to_csv(f"kpi_report_{best_method}.csv")
            
        except Exception as e:
            print(f"Error running KPI analysis: {e}")
    else:
        print("\n❌ No feasible solutions found for KPI analysis")

if __name__ == "__main__":
    # print("🚀 Starting comprehensive optimization comparison...")
    # comparison_results, scheduler = compare()
    
    # if comparison_results:
    #     print("\n🔍 Running KPI analysis on best solution...")
    #     run_kpi_analysis_on_best_solution(comparison_results)
        
    #     print("\n✅ Analysis complete! Check the generated files:")
    #     print("  • harmonogram_[method].csv - Individual schedules")
    #     print("  • optimization_comparison.csv - Comparison summary")
    #     print("  • kpi_report_[best_method].csv - KPI analysis")
    #     print("  • delay_histogram.png - Delay visualization")
    #     print("  • worker_daily_utilization.png - Worker utilization")
    # else:
    #     print("❌ Comparison failed. Please check your input files and try again.")
    scheduler, result = example_usage()