---

## **1. Cel projektu i struktura**

System ma dwa zasadnicze komponenty:

* **Moduł harmonogramowania (`harmv4.py`)**: tworzy optymalny plan realizacji zadań, przypisując je do dostępnych pracowników, uwzględniając m.in. ograniczenia czasowe, dostępność pracowników, oraz limity równoległości (`max_parallel`).
* **Moduł analizy KPI (`kpi.py`)**: na podstawie wygenerowanego harmonogramu ocenia realizację planu, mierząc m.in. terminowość, wykorzystanie zasobów i długość realizacji zadań.

## **2. Działanie modułu harmonogramowania – przegląd funkcjonalny**

### **2.1. Model decyzyjny**

* Każde zadanie ma przypisaną liczbę roboczogodzin do wykonania (`duration`) oraz dopuszczalny przedział czasowy (start–deadline).
* Zadania mogą być wykonywane równolegle przez wielu pracowników (do `max_parallel`), co jest kluczowym założeniem wpływającym na liczbę potrzebnych dni realizacji.
* Pracownicy mają ograniczenia dzienne (maks. 8h/dzień) i mogą mieć okresy niedostępności.

### **2.2. Ograniczenia i założenia modelu**

* Każde zadanie **musi być przypisane** do tylu pracowników, ile dopuszcza `max_parallel`, co w pewnych scenariuszach może sztucznie zwiększać trudność modelu (np. brak elastyczności w doborze liczby osób).
* Model zakłada **ciągłość realizacji zadań**, co redukuje możliwość rozciągania zadań w czasie w sposób nieregularny – to może kolidować z elastycznym zarządzaniem dostępnością.
* Zdefiniowane są constraints dotyczące:

  * dostępności pracowników,
  * braku przekroczenia dziennego limitu godzin,
  * niedopuszczalności pracy poza dniami roboczymi.

### **2.3. Funkcja celu**

* W analizowanym przypadku wykorzystano tryb `pack_jobs`, który premiuje „upakowanie” zadań w czasie, minimalizując rozproszenie godzin pracy i opóźnienia. Jest to podejście efektywnościowe, ale potencjalnie kolidujące z priorytetami terminowości.

### **2.4. Eksport danych**

* Wygenerowany harmonogram trafia do `harmonogram.csv` w postaci pracy dziennej: data–pracownik–zadanie–czas, co jest kluczowe dla dalszej ewaluacji KPI.


## 🔧 **1. Przygotowanie danych wejściowych**

### **1.1. Plik `jobs.csv` – lista zadań**

Format: `job_id;duration;start_date;deadline;max_parallel`
**Opis:**

* `job_id` – unikalny identyfikator zadania (np. Job1)
* `duration` – czas trwania w godzinach (np. 24)
* `start_date` – najwcześniejszy możliwy dzień rozpoczęcia (`YYYY-MM-DD`)
* `deadline` – ostateczny termin zakończenia (`YYYY-MM-DD`)
* `max_parallel` – maksymalna liczba pracowników pracujących równolegle nad zadaniem

**Przykład:**

```
job_id;duration;start_date;deadline;max_parallel
Job1;24;2025-01-15;2025-01-17;1
Job2;16;2025-01-15;2025-01-16;2
```


### **1.2. Plik `workers.csv` – dostępność pracowników**

Format: `worker_id;unavailable_start;unavailable_end`
**Opis:**

* `worker_id` – identyfikator pracownika
* `unavailable_start`, `unavailable_end` – okresy niedostępności (można zostawić puste)

**Przykład:**

```
worker_id;unavailable_start;unavailable_end
Worker_A;;
Worker_B;2025-01-16;2025-01-16
```


## 🧮 **2. Generowanie harmonogramu (`harmv4.py`)**

**Kroki:**

1. Upewnij się, że pliki `jobs.csv` i `workers.csv` są poprawnie przygotowane.
2. Uruchom skrypt `harmv4.py`, który:

   * wczytuje dane,
   * buduje model optymalizacyjny,
   * rozwiązuje problem (domyślnie z celem „pack\_jobs”),
   * zapisuje wynik do `harmonogram.csv`.

**Format wyjściowego `harmonogram.csv`:**

```
data;zadanie;czas;pracownik;uwagi
2025-01-15;Job1;8;Worker_A;
2025-01-16;Job2;8;Worker_B;Po terminie
```



## 📊 **3. Obliczanie KPI (`kpi.py`)**

**Kroki:**

1. Upewnij się, że `harmonogram.csv` i `jobs.csv` są w katalogu.
2. Uruchom `kpi.py`, który:

   * analizuje terminowość i wykorzystanie pracowników,
   * tworzy wykresy (`delay_histogram.png`, `worker_daily_utilization.png`),
   * generuje `kpi_report.csv` z czytelnymi wskaźnikami.

**Fragment przykładowego `kpi_report.csv`:**

```
KPI;Value
Percentage of tasks completed on time;0.00%
Average delay for late tasks;2.33 all days
Utilization of Worker_A;25.00%
```



## ✅ **Podsumowanie – wymagania i zalecenia**

* Wszystkie pliki CSV muszą mieć separator `;`.
* Format dat: `YYYY-MM-DD`.
* System jest wrażliwy na jakość danych – niekompletne lub niespójne dane mogą uniemożliwić wygenerowanie harmonogramu.



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
