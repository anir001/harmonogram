---
# 📅 System Harmonogramowania i Analizy KPI

Projekt ten służy do automatycznego tworzenia harmonogramów zadań produkcyjnych oraz oceny ich realizacji przy użyciu wskaźników KPI. Składa się z dwóch modułów:

* **harmv4.py** – optymalizator harmonogramu wykorzystujący OR-Tools (CP-SAT).
* **kpi.py** – moduł analizy wskaźników efektywności i terminowości zadań.



## ⚙️ Wymagania

* Python 3.8+
* Pakiety: `pandas`, `numpy`, `matplotlib`, `ortools`

Instalacja zależności:

```bash
pip install -r requirements.txt
```



## 📁 Struktura plików CSV

### `jobs.csv` – definicje zadań

| Kolumna        | Opis                                                  |
| -------------- | ----------------------------------------------------- |
| `job_id`       | Unikalny identyfikator zadania                        |
| `duration`     | Czas trwania zadania w godzinach                      |
| `start_date`   | Najwcześniejsza data rozpoczęcia (format: YYYY-MM-DD) |
| `deadline`     | Ostateczny termin zakończenia                         |
| `max_parallel` | Maks. liczba pracowników równolegle przy zadaniu      |

**Przykład:**

```csv
job_id;duration;start_date;deadline;max_parallel
Job1;24;2025-01-15;2025-01-17;1
Job2;16;2025-01-15;2025-01-16;2
```



### `workers.csv` – dostępność pracowników (opcjonalne)

| Kolumna             | Opis                         |
| ------------------- | ---------------------------- |
| `worker_id`         | Identyfikator pracownika     |
| `unavailable_start` | Data początku niedostępności |
| `unavailable_end`   | Data końca niedostępności    |

**Przykład:**

```csv
worker_id;unavailable_start;unavailable_end
Worker_A;;
Worker_B;2025-01-16;2025-01-16
```



## 🚀 Uruchamianie harmonogramu

Plik `harmv4.py`:

```bash
python harmv4.py
```

### Wyniki:

* Harmonogram zostanie zapisany do pliku `harmonogram.csv`:

```csv
data;zadanie;czas;pracownik;uwagi
2025-01-15;Job1;8;Worker_A;
2025-01-16;Job2;8;Worker_B;Po terminie
```


## 📊 Obliczanie KPI

Plik `kpi.py`:

```bash
python kpi.py
```

### Wygenerowane dane:

* `kpi_report.csv` – zestawienie wskaźników efektywności
* `delay_histogram.png` – histogram opóźnień
* `worker_daily_utilization.png` – wykorzystanie pracowników w czasie

**Przykład `kpi_report.csv`:**

```csv
KPI;Value
Percentage of tasks completed on time;0.00%
Average delay for late tasks;2.33 all days
Percentage of hours completed on time;16.67%
Utilization of Worker_A;25.00%
Utilization of Worker_B;50.00%
Utilization of Worker_C;75.00%
```


## 📌 Uwagi

* Wszystkie pliki CSV muszą używać średnika `;` jako separatora.
* Format dat: `YYYY-MM-DD`.
* System działa wyłącznie w dni robocze (poniedziałek–piątek).
* Niepoprawne lub niespójne dane mogą skutkować brakiem rozwiązania.

## ⚙️ Działanie modułu harmonogramowania – przegląd funkcjonalny**

### Model decyzyjny**

* Każde zadanie ma przypisaną liczbę roboczogodzin do wykonania (`duration`) oraz dopuszczalny przedział czasowy (start–deadline).
* Zadania mogą być wykonywane równolegle przez wielu pracowników (do `max_parallel`), co jest kluczowym założeniem wpływającym na liczbę potrzebnych dni realizacji.
* Pracownicy mają ograniczenia dzienne (maks. 8h/dzień) i mogą mieć okresy niedostępności.

### Ograniczenia i założenia modelu**

* Każde zadanie **musi być przypisane** do tylu pracowników, ile dopuszcza `max_parallel`.
* Model zakłada **ciągłość realizacji zadań**, co redukuje możliwość rozciągania zadań w czasie w sposób nieregularny.
* Zdefiniowane są wymagania dotyczące:

  * dostępności pracowników,
  * braku przekroczenia dziennego limitu godzin,
  * niedopuszczalności pracy poza dniami roboczymi.

