import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import os

class DiabetesPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система предсказания диабета (Excel/CSV)")
        self.root.geometry("1200x800")
        
        self.model = None
        self.scaler = None
        self.data = None
        self.features = None
        self.history = []
        self.current_file = None
        self.file_type = None

        # Настройки темы
        self.dark_mode = False
        self.bg_color = "#f0f0f0"
        self.fg_color = "#000000"
        self.button_bg = "#e0e0e0"

        # Создание основного интерфейса
        self.create_widgets()

        # Меню
        self.create_menu()

        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Готово")
        status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_menu(self):
        menubar = tk.Menu(self.root)

        # Меню файла
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Открыть Excel файл", command=lambda: self.load_data_file('excel'))
        file_menu.add_command(label="Открыть CSV файл", command=lambda: self.load_data_file('csv'))
        file_menu.add_command(label="Сохранить данные", command=self.save_data)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)

        # Меню модели
        model_menu = tk.Menu(menubar, tearoff=0)
        model_menu.add_command(label="Обучить модель", command=self.train_model)
        model_menu.add_command(label="Оценка модели", command=self.evaluate_model)
        menubar.add_cascade(label="Модель", menu=model_menu)
        
        # Меню настроек
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Темная тема", command=self.toggle_dark_mode)
        menubar.add_cascade(label="Настройки", menu=settings_menu)
        
        self.root.config(menu=menubar)
    
    def create_widgets(self):
        # Создание вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Вкладка данных
        self.create_data_tab()

        # Вкладка прогнозирования
        self.create_prediction_tab()

        # Вкладка визуализации
        self.create_visualization_tab()

        # Вкладка истории
        self.create_history_tab()
    
    def create_data_tab(self):
        self.data_tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.data_tab, text="Данные")

        # Кнопки управления данными
        btn_frame = tk.Frame(self.data_tab, bg=self.bg_color)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        load_excel_btn = tk.Button(btn_frame, text="Загрузить Excel", 
                                 command=lambda: self.load_data_file('excel'), bg=self.button_bg)
        load_excel_btn.pack(side=tk.LEFT, padx=5)
        
        load_csv_btn = tk.Button(btn_frame, text="Загрузить CSV", 
                               command=lambda: self.load_data_file('csv'), bg=self.button_bg)
        load_csv_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = tk.Button(btn_frame, text="Сохранить данные", command=self.save_data, bg=self.button_bg)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        train_btn = tk.Button(btn_frame, text="Обучить модель", command=self.train_model, bg=self.button_bg)
        train_btn.pack(side=tk.LEFT, padx=5)

        # Treeview для отображения данных
        self.tree_frame = tk.Frame(self.data_tab)
        self.tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tree = ttk.Treeview(self.tree_frame)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Фрейм для добавления/редактирования данных
        self.edit_frame = tk.LabelFrame(self.data_tab, text="Добавить/Редактировать данные", 
                                      bg=self.bg_color, fg=self.fg_color)
        self.edit_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.data_entries = {}
        self.add_edit_controls()
    
    def add_edit_controls(self):
        # Очистка предыдущих элементов
        for widget in self.edit_frame.winfo_children():
            widget.destroy()
        
        if self.features:
            # Создание полей ввода для каждого признака
            for i, feature in enumerate(self.features):
                row = i // 4
                col = i % 4
                
                frame = tk.Frame(self.edit_frame, bg=self.bg_color)
                frame.grid(row=row, column=col, padx=5, pady=5, sticky="w")
                
                label = tk.Label(frame, text=f"{feature}:", bg=self.bg_color, fg=self.fg_color)
                label.pack(side=tk.LEFT)
                
                entry = tk.Entry(frame, width=12)
                entry.pack(side=tk.LEFT)
                self.data_entries[feature] = entry

            # Кнопки управления
            btn_frame = tk.Frame(self.edit_frame, bg=self.bg_color)
            btn_frame.grid(row=(len(self.features) // 4) + 1, column=0, columnspan=4, pady=5)
            
            add_btn = tk.Button(btn_frame, text="Добавить", command=self.add_data, bg=self.button_bg)
            add_btn.pack(side=tk.LEFT, padx=5)
            
            update_btn = tk.Button(btn_frame, text="Обновить", command=self.update_data, bg=self.button_bg)
            update_btn.pack(side=tk.LEFT, padx=5)
            
            delete_btn = tk.Button(btn_frame, text="Удалить", command=self.delete_data, bg=self.button_bg)
            delete_btn.pack(side=tk.LEFT, padx=5)
            
            clear_btn = tk.Button(btn_frame, text="Очистить", command=self.clear_entries, bg=self.button_bg)
            clear_btn.pack(side=tk.LEFT, padx=5)
    
    def create_prediction_tab(self):
        self.prediction_tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.prediction_tab, text="Прогнозирование")

        # Фрейм для ввода данных
        input_frame = tk.LabelFrame(self.prediction_tab, text="Введите данные пациента", 
                                  bg=self.bg_color, fg=self.fg_color)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.pred_entries = {}
        if self.features:
            for i, feature in enumerate(self.features):
                row = i // 3
                col = i % 3
                
                frame = tk.Frame(input_frame, bg=self.bg_color)
                frame.grid(row=row, column=col, padx=5, pady=5, sticky="w")
                
                label = tk.Label(frame, text=f"{feature}:", bg=self.bg_color, fg=self.fg_color)
                label.pack(side=tk.LEFT)
                
                entry = tk.Entry(frame, width=15)
                entry.pack(side=tk.LEFT)
                self.pred_entries[feature] = entry

        # Кнопка прогнозирования
        predict_btn = tk.Button(self.prediction_tab, text="Сделать прогноз", 
                              command=self.make_prediction, bg=self.button_bg)
        predict_btn.pack(pady=10)

        # Фрейм результата
        result_frame = tk.LabelFrame(self.prediction_tab, text="Результат", 
                                   bg=self.bg_color, fg=self.fg_color)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.result_text = tk.Text(result_frame, height=10, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Кнопка экспорта
        export_btn = tk.Button(self.prediction_tab, text="Экспорт в PDF", 
                             command=self.export_result_to_pdf, bg=self.button_bg)
        export_btn.pack(pady=5)
    
    def create_visualization_tab(self):
        self.visualization_tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.visualization_tab, text="Визуализация")

        # Управление визуализацией
        control_frame = tk.Frame(self.visualization_tab, bg=self.bg_color)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(control_frame, text="Тип графика:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT)
        self.plot_type = tk.StringVar(value="scatter")
        plot_menu = ttk.OptionMenu(control_frame, self.plot_type, "scatter", "scatter", "histogram", "boxplot")
        plot_menu.pack(side=tk.LEFT, padx=5)
        
        tk.Label(control_frame, text="X:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT)
        self.x_axis = tk.StringVar()
        x_dropdown = ttk.Combobox(control_frame, textvariable=self.x_axis, width=15)
        x_dropdown.pack(side=tk.LEFT, padx=5)
        
        tk.Label(control_frame, text="Y:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT)
        self.y_axis = tk.StringVar()
        y_dropdown = ttk.Combobox(control_frame, textvariable=self.y_axis, width=15)
        y_dropdown.pack(side=tk.LEFT, padx=5)
        
        plot_btn = tk.Button(control_frame, text="Построить", command=self.update_plot, bg=self.button_bg)
        plot_btn.pack(side=tk.LEFT, padx=10)
        
        # Фрейм для графика
        self.plot_frame = tk.Frame(self.visualization_tab, bg=self.bg_color)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Инициализация пустого графика
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_title("Выберите данные для визуализации")
        self.canvas.draw()
    
    def create_history_tab(self):
        self.history_tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.history_tab, text="История")
        
        # Treeview для истории
        self.history_tree = ttk.Treeview(self.history_tab, columns=("Дата", "Входные данные", "Результат"))
        self.history_tree.heading("#0", text="ID")
        self.history_tree.heading("Дата", text="Дата")
        self.history_tree.heading("Входные данные", text="Входные данные")
        self.history_tree.heading("Результат", text="Результат")
        
        self.history_tree.column("#0", width=50)
        self.history_tree.column("Дата", width=150)
        self.history_tree.column("Входные данные", width=400)
        self.history_tree.column("Результат", width=200)
        
        scrollbar = ttk.Scrollbar(self.history_tab, orient="vertical", command=self.history_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        
        # Кнопки управления историей
        btn_frame = tk.Frame(self.history_tab, bg=self.bg_color)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        clear_btn = tk.Button(btn_frame, text="Очистить историю", command=self.clear_history, bg=self.button_bg)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        export_btn = tk.Button(btn_frame, text="Экспорт истории", command=self.export_history, bg=self.button_bg)
        export_btn.pack(side=tk.LEFT, padx=5)
    
    def load_data_file(self, file_type):
        """Загружает данные из файла (Excel или CSV)"""
        filetypes = []
        if file_type == 'excel':
            filetypes = [("Excel files", "*.xlsx *.xls")]
            title = "Открыть Excel файл"
        elif file_type == 'csv':
            filetypes = [("CSV files", "*.csv")]
            title = "Открыть CSV файл"
        
        file_path = filedialog.askopenfilename(filetypes=filetypes, title=title)
        if file_path:
            try:
                self.current_file = file_path
                self.file_type = file_type
                
                if file_type == 'excel':
                    self.data = pd.read_excel(file_path)
                else:  # CSV
                    self.data = pd.read_csv(file_path)
                
                # Проверяем наличие целевой переменной
                if 'Diabetes_012' not in self.data.columns:
                    messagebox.showerror("Ошибка", "Файл должен содержать столбец 'Diabetes_012'")
                    return
                
                self.features = [col for col in self.data.columns if col != 'Diabetes_012']
                
                # Обновление интерфейса
                self.update_data_display()
                self.update_prediction_controls()
                self.update_visualization_controls()
                
                self.status_var.set(f"Загружен файл: {os.path.basename(file_path)}")
                messagebox.showinfo("Успех", "Файл успешно загружен")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")
    
    def save_data(self):
        if self.data is None:
            messagebox.showerror("Ошибка", "Нет данных для сохранения")
            return
        
        if self.current_file:
            default_ext = os.path.splitext(self.current_file)[1]
            default_path = self.current_file
        else:
            default_ext = '.xlsx' if self.file_type == 'excel' else '.csv'
            default_path = f"diabetes_data{default_ext}"
            
        filetypes = []
        if self.file_type == 'excel':
            filetypes = [("Excel files", "*.xlsx")]
            defaultextension = ".xlsx"
        else:
            filetypes = [("CSV files", "*.csv")]
            defaultextension = ".csv"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=defaultextension,
            filetypes=filetypes,
            initialfile=os.path.basename(default_path)
        )
        
        if file_path:
            try:
                if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                    self.data.to_excel(file_path, index=False)
                    self.file_type = 'excel'
                else:
                    self.data.to_csv(file_path, index=False)
                    self.file_type = 'csv'
                
                self.current_file = file_path
                self.status_var.set(f"Данные сохранены в: {os.path.basename(file_path)}")
                messagebox.showinfo("Успех", "Данные успешно сохранены")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")
    
    def update_data_display(self):
        # Очистка Treeview
        self.tree.delete(*self.tree.get_children())
        
        # Настройка колонок
        self.tree["columns"] = list(self.data.columns)
        for col in self.data.columns:
            self.tree.column(col, width=100, anchor=tk.CENTER)
            self.tree.heading(col, text=col)
        
        # Добавление данных (первые 100 строк для производительности)
        for i, row in self.data.head(100).iterrows():
            self.tree.insert("", "end", values=list(row))
        
        # Обновление полей ввода
        self.add_edit_controls()
    
    def update_prediction_controls(self):
        # Очистка предыдущих полей ввода
        for widget in self.prediction_tab.winfo_children():
            if isinstance(widget, tk.LabelFrame) and widget.winfo_name() == "!labelframe":
                widget.destroy()
                break
        
        # Создание новых полей ввода
        input_frame = tk.LabelFrame(self.prediction_tab, text="Введите данные пациента", 
                                  bg=self.bg_color, fg=self.fg_color)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.pred_entries = {}
        if self.features:
            for i, feature in enumerate(self.features):
                row = i // 3
                col = i % 3
                
                frame = tk.Frame(input_frame, bg=self.bg_color)
                frame.grid(row=row, column=col, padx=5, pady=5, sticky="w")
                
                label = tk.Label(frame, text=f"{feature}:", bg=self.bg_color, fg=self.fg_color)
                label.pack(side=tk.LEFT)
                
                entry = tk.Entry(frame, width=15)
                entry.pack(side=tk.LEFT)
                self.pred_entries[feature] = entry
    
    def update_visualization_controls(self):
        if self.features:
            self.x_axis.set(self.features[0])
            self.y_axis.set(self.features[1] if len(self.features) > 1 else "")
            
            x_dropdown = self.visualization_tab.children['!combobox']
            y_dropdown = self.visualization_tab.children['!combobox2']
            
            x_dropdown['values'] = self.features
            y_dropdown['values'] = self.features
    
    def train_model(self):
        if self.data is None:
            messagebox.showerror("Ошибка", "Сначала загрузите данные")
            return
        
        try:
            X = self.data[self.features]
            y = self.data['Diabetes_012']
            
            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Масштабирование
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Обучение модели
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            # Оценка на тестовых данных
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.status_var.set(f"Модель обучена. Точность: {accuracy:.2f}")
            messagebox.showinfo("Успех", f"Модель успешно обучена. Точность: {accuracy:.2f}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обучении модели: {str(e)}")
    
    def evaluate_model(self):
        if self.model is None or self.data is None:
            messagebox.showerror("Ошибка", "Сначала обучите модель")
            return
        
        try:
            X = self.data[self.features]
            y = self.data['Diabetes_012']
            
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_test_scaled = self.scaler.transform(X_test)
            
            y_pred = self.model.predict(X_test_scaled)
            report = classification_report(y_test, y_pred)
            
            # Отображение отчета
            report_window = tk.Toplevel(self.root)
            report_window.title("Оценка модели")
            
            text = tk.Text(report_window, wrap=tk.WORD)
            text.pack(fill=tk.BOTH, expand=True)
            
            text.insert(tk.END, report)
            text.config(state=tk.DISABLED)
            
            self.status_var.set("Модель оценена. См. отчет.")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при оценке модели: {str(e)}")
    
    def make_prediction(self):
        if self.model is None:
            messagebox.showerror("Ошибка", "Сначала обучите модель")
            return
        
        try:
            # Сбор данных из полей ввода
            input_data = {}
            for feature, entry in self.pred_entries.items():
                value = entry.get()
                if not value:
                    raise ValueError(f"Пожалуйста, заполните поле {feature}")
                input_data[feature] = float(value)
            
            # Преобразование в DataFrame
            input_df = pd.DataFrame([input_data])[self.features]
            
            # Масштабирование
            scaled_data = self.scaler.transform(input_df)
            
            # Прогнозирование
            prediction = self.model.predict(scaled_data)[0]
            proba = self.model.predict_proba(scaled_data)[0]
            
            # Интерпретация результата
            if prediction == 0:
                result = "Нет диабета"
            elif prediction == 1:
                result = "Предиабет"
            else:
                result = "Диабет"
            
            # Отображение результата
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Результат прогноза: {result}\n\n")
            self.result_text.insert(tk.END, f"Вероятности:\n")
            self.result_text.insert(tk.END, f"- Нет диабета: {proba[0]*100:.2f}%\n")
            self.result_text.insert(tk.END, f"- Предиабет: {proba[1]*100:.2f}%\n")
            self.result_text.insert(tk.END, f"- Диабет: {proba[2]*100:.2f}%\n")
            
            # Добавление в историю
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.history.append({
                "timestamp": timestamp,
                "input_data": input_data,
                "result": result,
                "probabilities": proba
            })
            
            self.update_history_display()
            self.status_var.set("Прогноз выполнен")
            
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при прогнозировании: {str(e)}")
    
    def update_history_display(self):
        self.history_tree.delete(*self.history_tree.get_children())
        
        for i, record in enumerate(self.history[-50:]):  # Показываем последние 50 записей
            input_str = ", ".join(f"{k}: {v}" for k, v in record['input_data'].items())
            self.history_tree.insert("", "end", text=str(i+1), 
                                   values=(record['timestamp'], input_str[:100] + "..." if len(input_str) > 100 else input_str, 
                                   record['result']))
    
    def clear_history(self):
        self.history = []
        self.history_tree.delete(*self.history_tree.get_children())
        self.status_var.set("История очищена")
    
    def export_history(self):
        if not self.history:
            messagebox.showerror("Ошибка", "История пуста")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            initialfile="diabetes_predictions_history.xlsx"
        )
        
        if file_path:
            try:
                history_df = pd.DataFrame(self.history)
                if file_path.endswith('.csv'):
                    history_df.to_csv(file_path, index=False)
                else:
                    history_df.to_excel(file_path, index=False)
                
                self.status_var.set(f"История экспортирована в: {os.path.basename(file_path)}")
                messagebox.showinfo("Успех", "История успешно экспортирована")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось экспортировать историю: {str(e)}")
    
    def update_plot(self):
        if self.data is None:
            return
        
        plot_type = self.plot_type.get()
        x_feature = self.x_axis.get()
        
        if not x_feature:
            messagebox.showerror("Ошибка", "Выберите признак для оси X")
            return
        
        self.ax.clear()
        
        try:
            if plot_type == "scatter":
                y_feature = self.y_axis.get()
                if not y_feature:
                    messagebox.showerror("Ошибка", "Для диаграммы рассеяния выберите признак для оси Y")
                    return
                
                sns.scatterplot(data=self.data, x=x_feature, y=y_feature, hue='Diabetes_012', ax=self.ax)
                self.ax.set_title(f"Диаграмма рассеяния: {x_feature} vs {y_feature}")
            
            elif plot_type == "histogram":
                sns.histplot(data=self.data, x=x_feature, hue='Diabetes_012', kde=True, ax=self.ax)
                self.ax.set_title(f"Гистограмма распределения {x_feature}")
            
            elif plot_type == "boxplot":
                sns.boxplot(data=self.data, x='Diabetes_012', y=x_feature, ax=self.ax)
                self.ax.set_title(f"Ящик с усами для {x_feature} по классам диабета")
            
            self.ax.grid(True)
            self.canvas.draw()
            self.status_var.set(f"Построен график: {plot_type}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось построить график: {str(e)}")
    
    def add_data(self):
        if self.data is None:
            messagebox.showerror("Ошибка", "Сначала загрузите данные")
            return
        
        try:
            new_data = {}
            for feature, entry in self.data_entries.items():
                value = entry.get()
                if not value:
                    raise ValueError(f"Пожалуйста, заполните поле {feature}")
                new_data[feature] = float(value)
            
            # Добавление Diabetes_012 (по умолчанию -1, можно изменить)
            new_data['Diabetes_012'] = -1
            
            # Добавление в DataFrame
            new_row = pd.DataFrame([new_data])
            self.data = pd.concat([self.data, new_row], ignore_index=True)
            
            # Обновление отображения
            self.update_data_display()
            self.clear_entries()
            self.status_var.set("Данные добавлены")
            
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при добавлении данных: {str(e)}")
    
    def update_data(self):
        if self.data is None:
            messagebox.showerror("Ошибка", "Сначала загрузите данные")
            return
        
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showerror("Ошибка", "Выберите строку для обновления")
            return
        
        try:
            # Получение индекса выбранной строки
            item = self.tree.item(selected_item[0])
            index = self.tree.index(selected_item[0])
            
            # Обновление данных
            for feature, entry in self.data_entries.items():
                value = entry.get()
                if not value:
                    raise ValueError(f"Пожалуйста, заполните поле {feature}")
                self.data.at[index, feature] = float(value)
            
            # Обновление отображения
            self.update_data_display()
            self.clear_entries()
            self.status_var.set(f"Строка {index} обновлена")
            
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обновлении данных: {str(e)}")
    
    def delete_data(self):
        if self.data is None:
            messagebox.showerror("Ошибка", "Сначала загрузите данные")
            return
        
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showerror("Ошибка", "Выберите строку для удаления")
            return
        
        try:
            # Получение индекса выбранной строки
            index = self.tree.index(selected_item[0])
            
            # Удаление строки
            self.data = self.data.drop(index).reset_index(drop=True)
            
            # Обновление отображения
            self.update_data_display()
            self.clear_entries()
            self.status_var.set(f"Строка {index} удалена")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при удалении данных: {str(e)}")
    
    def clear_entries(self):
        for entry in self.data_entries.values():
            entry.delete(0, tk.END)
    
    def export_result_to_pdf(self):
        if not self.history:
            messagebox.showerror("Ошибка", "Нет результатов для экспорта")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile="diabetes_prediction_report.pdf"
        )
        
        if file_path:
            try:
                last_record = self.history[-1]
                
                c = canvas.Canvas(file_path, pagesize=letter)
                width, height = letter
                
                # Заголовок
                c.setFont("Helvetica-Bold", 16)
                c.drawString(100, height - 50, "Отчет о прогнозе диабета")
                
                # Информация о файле
                c.setFont("Helvetica", 10)
                if self.current_file:
                    c.drawString(100, height - 80, f"Источник данных: {os.path.basename(self.current_file)}")
                
                # Дата и время
                c.drawString(100, height - 100, f"Дата прогноза: {last_record['timestamp']}")
                
                # Результат
                c.setFont("Helvetica-Bold", 14)
                c.drawString(100, height - 130, f"Результат: {last_record['result']}")
                
                # Вероятности
                c.setFont("Helvetica", 12)
                c.drawString(100, height - 160, "Вероятности:")
                c.drawString(120, height - 180, f"- Нет диабета: {last_record['probabilities'][0]*100:.2f}%")
                c.drawString(120, height - 200, f"- Предиабет: {last_record['probabilities'][1]*100:.2f}%")
                c.drawString(120, height - 220, f"- Диабет: {last_record['probabilities'][2]*100:.2f}%")
                
                # Входные данные
                c.drawString(100, height - 250, "Входные данные:")
                y = height - 270
                for k, v in last_record['input_data'].items():
                    c.drawString(120, y, f"{k}: {v}")
                    y -= 20
                    if y < 100:
                        c.showPage()
                        y = height - 50
                        c.setFont("Helvetica", 12)
                
                c.save()
                self.status_var.set(f"Отчет экспортирован в: {os.path.basename(file_path)}")
                messagebox.showinfo("Успех", "Отчет успешно экспортирован")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось экспортировать отчет: {str(e)}")
    
    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        
        if self.dark_mode:
            self.bg_color = "#2d2d2d"
            self.fg_color = "#ffffff"
            self.button_bg = "#4d4d4d"
        else:
            self.bg_color = "#f0f0f0"
            self.fg_color = "#000000"
            self.button_bg = "#e0e0e0"
        
        self.update_widget_colors()
    
    def update_widget_colors(self):
        # Обновление цветов всех виджетов
        widgets = [self.data_tab, self.prediction_tab, self.visualization_tab, self.history_tab]
        
        for widget in widgets:
            widget.config(bg=self.bg_color)
            
            # Рекурсивное обновление дочерних виджетов
            for child in widget.winfo_children():
                self.update_child_colors(child)
        
        # Обновление текстовых виджетов
        self.result_text.config(bg="white" if not self.dark_mode else "#333333", 
                             fg="black" if not self.dark_mode else "white")
    
    def update_child_colors(self, widget):
        if isinstance(widget, (tk.Frame, tk.LabelFrame)):
            widget.config(bg=self.bg_color)
        
        if isinstance(widget, (tk.Label, tk.Button)):
            widget.config(bg=self.bg_color, fg=self.fg_color)
        
        if isinstance(widget, tk.Button):
            widget.config(bg=self.button_bg)
        
        if isinstance(widget, tk.Text):
            widget.config(bg="white" if not self.dark_mode else "#333333", 
                         fg="black" if not self.dark_mode else "white")
        
        for child in widget.winfo_children():
            self.update_child_colors(child)

# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesPredictorApp(root)
    root.mainloop()
