import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from scipy import stats
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Clasificación de Atletas", 
    page_icon="🏃", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("Sistema de Clasificación de Atletas 🏃‍♂️")

# Sidebar navigation
st.sidebar.title("Navegación")
page = st.sidebar.radio(
    "Selecciona una página:", 
    ["🏠 Inicio", "🔍 Preprocesamiento", "📊 Modelado", "📈 Métricas", "🔮 Predicción"]
)

# Constantes
DATA_PATH = "deportistas.csv"
MODEL_DT_PATH = "modelo_dt.pkl"
MODEL_LR_PATH = "modelo_lr.pkl"
METRICS_PATH = "metricas_modelo.txt"

# Funciones de utilidad
@st.cache_data
def load_data():
    """Carga los datos y realiza limpieza inicial"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Seleccionar solo las variables requeridas
        required_cols = [
            "frecuencia_cardiaca_basal", 
            "porcentaje_fibras_lentas", 
            "porcentaje_fibras_rapidas", 
            "indice_masa_corporal", 
            "tipo_atleta"
        ]
        
        # Verificar que las columnas requeridas existan
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Columnas faltantes en el dataset: {missing_cols}")
            return pd.DataFrame()
        
        df = df[required_cols]
        
        # Introducir algunos NaN y outliers artificialmente para demostración
        if st.session_state.get('data_loaded', False) == False:
            # Introducir NaN aleatorios (5% de los datos)
            mask = np.random.random(df.shape) < 0.05
            df = df.mask(mask)
            
            # Introducir outliers en algunas columnas
            outlier_mask = np.random.random(df.shape) < 0.03
            outlier_values = df.select_dtypes(include=['float64']).apply(
                lambda x: x.mean() + 5 * x.std() * np.random.randn(len(x))
            )
            df = df.mask(outlier_mask, outlier_values)
            
            st.session_state['data_loaded'] = True
        
        return df
    
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return pd.DataFrame()

def save_model(model, path):
    """Guarda un modelo en disco"""
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path):
    """Carga un modelo desde disco"""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def save_metrics(metrics):
    """Guarda las métricas en un archivo"""
    with open(METRICS_PATH, "w") as f:
        f.write(metrics)

def load_metrics():
    """Carga las métricas desde un archivo"""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return f.read()
    return None

# Página de Inicio
if page == "🏠 Inicio":
    st.markdown("""
    ## Bienvenido al Sistema de Clasificación de Atletas
    
    Esta aplicación permite clasificar atletas en dos categorías:
    - **Fondistas**: Atletas de resistencia
    - **Velocistas**: Atletas de velocidad
    
    ### Funcionalidades:
    
    1. **🔍 Preprocesamiento**: Análisis y limpieza de datos
    2. **📊 Modelado**: Entrenamiento de modelos de clasificación
    3. **📈 Métricas**: Evaluación y comparación de modelos
    4. **🔮 Predicción**: Clasificación de nuevos atletas
    
    ### Datos utilizados:
    - Frecuencia cardíaca basal
    - Porcentaje de fibras lentas
    - Porcentaje de fibras rápidas
    - Índice de masa corporal (IMC)
    
    ### Modelos implementados:
    - Árboles de Decisión
    - Regresión Logística
    """)
    
    st.image("https://images.unsplash.com/photo-1552674605-db6ffd4facb5?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
             caption="Clasificación de Atletas", use_column_width=True)

# Página de Preprocesamiento
elif page == "🔍 Preprocesamiento":
    st.header("🔍 Preprocesamiento de Datos")
    
    # Cargar datos
    df = load_data()
    
    if df.empty:
        st.warning("No se pudieron cargar los datos. Verifica el archivo de datos.")
        st.stop()
    
    # Mostrar datos originales
    st.subheader("Datos Originales")
    st.write(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    st.dataframe(df.head())
    
    # Sección de valores faltantes
    st.subheader("1. Manejo de Valores Faltantes (NaN)")
    st.write("Valores faltantes por columna:")
    missing_data = df.isna().sum().to_frame("Valores Faltantes")
    st.dataframe(missing_data)
    
    # Opciones para manejar NaN
    nan_option = st.radio(
        "Estrategia para manejar valores faltantes:",
        ["Eliminar filas con NaN", "Imputar con la mediana"]
    )
    
    if st.button("Aplicar manejo de NaN"):
        if nan_option == "Eliminar filas con NaN":
            df_clean = df.dropna()
            st.success(f"Se eliminaron {len(df) - len(df_clean)} filas con valores faltantes.")
        else:
            imputer = SimpleImputer(strategy="median")
            df_clean = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['float64'])), 
                                  columns=df.select_dtypes(include=['float64']).columns)
            df_clean['tipo_atleta'] = df['tipo_atleta']  # Mantener la columna categórica
            st.success("Valores faltantes imputados con la mediana de cada columna.")
        
        st.session_state['df_clean'] = df_clean
        st.dataframe(df_clean.head())
    
    # Sección de outliers si los datos están limpios
    if 'df_clean' in st.session_state:
        df_clean = st.session_state['df_clean']
        
        st.subheader("2. Detección y Manejo de Outliers")
        
        # Seleccionar columnas numéricas
        numeric_cols = df_clean.select_dtypes(include=['float64']).columns
        
        # Mostrar visualización de outliers
        col_selected = st.selectbox("Selecciona una columna para analizar outliers:", numeric_cols)
        
        # Gráficos de outliers
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Boxplot
        sns.boxplot(data=df_clean, y=col_selected, ax=ax1)
        ax1.set_title(f'Boxplot de {col_selected}')
        
        # Histograma
        sns.histplot(data=df_clean, x=col_selected, kde=True, ax=ax2)
        ax2.set_title(f'Distribución de {col_selected}')
        
        st.pyplot(fig)
        
        # Detección cuantitativa de outliers
        st.write("**Detección cuantitativa de outliers (método Z-score > 3):**")
        z_scores = stats.zscore(df_clean[col_selected])
        outliers = df_clean[(abs(z_scores) > 3)]
        
        st.write(f"Número de outliers detectados en {col_selected}: {len(outliers)}")
        if not outliers.empty:
            st.dataframe(outliers.head())
        
        # Opciones para manejar outliers
        outlier_option = st.radio(
            "Estrategia para manejar outliers:",
            ["Mantener outliers", "Eliminar outliers", "Reemplazar con valores límite"]
        )
        
        if st.button("Aplicar manejo de outliers"):
            if outlier_option == "Eliminar outliers":
                df_no_outliers = df_clean[(abs(z_scores) <= 3)]
                st.success(f"Se eliminaron {len(df_clean) - len(df_no_outliers)} outliers.")
            elif outlier_option == "Reemplazar con valores límite":
                lower_limit = df_clean[col_selected].mean() - 3 * df_clean[col_selected].std()
                upper_limit = df_clean[col_selected].mean() + 3 * df_clean[col_selected].std()
                df_no_outliers = df_clean.copy()
                df_no_outliers[col_selected] = np.where(
                    df_no_outliers[col_selected] > upper_limit, upper_limit,
                    np.where(
                        df_no_outliers[col_selected] < lower_limit, lower_limit,
                        df_no_outliers[col_selected]
                    )
                )
                st.success("Outliers reemplazados con valores límite.")
            else:
                df_no_outliers = df_clean.copy()
                st.info("Outliers mantenidos en el dataset.")
            
            st.session_state['df_no_outliers'] = df_no_outliers
            st.dataframe(df_no_outliers.describe())
    
    # Sección de codificación si los datos están limpios y sin outliers
    if 'df_no_outliers' in st.session_state:
        df_final = st.session_state['df_no_outliers']
        
        st.subheader("3. Codificación de Variables Categóricas")
        
        # Mostrar valores únicos de la variable objetivo
        st.write("Valores únicos en 'tipo_atleta':", df_final['tipo_atleta'].unique())
        
        # Codificar variable objetivo
        le = LabelEncoder()
        df_final['tipo_atleta_encoded'] = le.fit_transform(df_final['tipo_atleta'])
        
        st.write("**Codificación:**")
        st.write(pd.DataFrame({
            'Valor Original': le.classes_,
            'Valor Codificado': range(len(le.classes_))
        }))
        
        st.session_state['df_final'] = df_final
        st.session_state['label_encoder'] = le
        
        # Visualización de distribuciones
        st.subheader("4. Distribución de Variables")
        
        # Distribución por tipo de atleta
        st.write("**Distribución por tipo de atleta:**")
        numeric_cols = df_final.select_dtypes(include=['float64']).columns
        
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(
                data=df_final, 
                x=col, 
                hue='tipo_atleta', 
                kde=True, 
                multiple='stack',
                palette='viridis',
                ax=ax
            )
            ax.set_title(f'Distribución de {col} por tipo de atleta')
            st.pyplot(fig)
        
        # Correlación y multicolinealidad
        st.subheader("5. Correlación y Multicolinealidad")
        
        # Matriz de correlación
        corr_matrix = df_final[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap="coolwarm", 
            center=0, 
            fmt=".2f",
            linewidths=.5,
            ax=ax
        )
        ax.set_title("Matriz de Correlación")
        st.pyplot(fig)
        
        # Detección de alta correlación
        st.write("**Análisis de Multicolinealidad:**")
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr:
            st.warning("**¡Alerta!** Se detectaron altas correlaciones entre:")
            for pair in high_corr:
                st.write(f"- {pair[0]} y {pair[1]}: {pair[2]:.2f}")
            st.info("""
            **Recomendación:** 
            Variables con alta correlación pueden causar multicolinealidad. 
            Considera eliminar una de las variables correlacionadas o usar técnicas como PCA.
            """)
        else:
            st.success("No se detectaron altas correlaciones entre variables predictoras.")
        
        # Balance de clases
        st.subheader("6. Balance de Clases")
        
        class_counts = df_final['tipo_atleta'].value_counts()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico de barras
        class_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
        ax1.set_title("Distribución de Clases")
        ax1.set_xticklabels(class_counts.index, rotation=0)
        ax1.set_ylabel("Cantidad")
        
        # Gráfico de pastel
        class_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2, colors=['skyblue', 'salmon'])
        ax2.set_title("Proporción de Clases")
        ax2.set_ylabel("")
        
        st.pyplot(fig)
        
        st.write(f"""
        - **Fondistas:** {class_counts[0]} ({class_counts[0]/len(df_final):.1%})
        - **Velocistas:** {class_counts[1]} ({class_counts[1]/len(df_final):.1%})
        """)
        
        if abs(class_counts[0] - class_counts[1]) > 0.2 * len(df_final):
            st.warning("""
            **¡Advertencia!** El dataset está desbalanceado. 
            Esto puede afectar el rendimiento del modelo, causando un sesgo hacia la clase mayoritaria.
            
            **Soluciones posibles:**
            - Usar técnicas de remuestreo (oversampling/undersampling)
            - Utilizar pesos de clase en el modelo
            - Emplear métricas apropiadas (F1-score, AUC-ROC)
            """)
        else:
            st.success("El dataset está razonablemente balanceado.")

# Página de Modelado
elif page == "📊 Modelado":
    st.header("📊 Entrenamiento de Modelos")
    
    if 'df_final' not in st.session_state:
        st.warning("Por favor, complete el preprocesamiento en la página anterior primero.")
        st.stop()
    
    df_final = st.session_state['df_final']
    
    st.subheader("Preparación de Datos")
    
    # Dividir datos en características (X) y objetivo (y)
    X = df_final.drop(columns=['tipo_atleta', 'tipo_atleta_encoded'])
    y = df_final['tipo_atleta_encoded']
    
    # Dividir en conjuntos de entrenamiento y prueba
    test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.3, 0.2, 0.05)
    random_state = st.number_input("Semilla aleatoria:", min_value=0, value=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    st.write(f"""
    - **Entrenamiento:** {X_train.shape[0]} muestras
    - **Prueba:** {X_test.shape[0]} muestras
    """)
    
    # Escalar datos para regresión logística
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.session_state['X_train_scaled'] = X_train_scaled
    st.session_state['X_test_scaled'] = X_test_scaled
    st.session_state['scaler'] = scaler
    
    # Configuración de modelos
    st.subheader("Configuración de Modelos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Árbol de Decisión**")
        max_depth = st.slider("Profundidad máxima", 1, 20, 5, key='dt_depth')
        min_samples_split = st.slider("Mínimo muestras para dividir", 2, 20, 2, key='dt_samples_split')
        criterion = st.selectbox("Criterio", ["gini", "entropy"], key='dt_criterion')
    
    with col2:
        st.markdown("**Regresión Logística**")
        C = st.slider("Parámetro de regularización (C)", 0.01, 10.0, 1.0, key='lr_C')
        penalty = st.selectbox("Tipo de regularización", ["l2", "l1"], key='lr_penalty')
        max_iter = st.slider("Máximo de iteraciones", 100, 1000, 100, key='lr_max_iter')
    
    # Entrenar modelos
    if st.button("Entrenar Modelos"):
        with st.spinner("Entrenando modelos..."):
            # Entrenar Árbol de Decisión
            dt_model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
                random_state=random_state
            )
            dt_model.fit(X_train, y_train)
            
            # Entrenar Regresión Logística
            lr_model = LogisticRegression(
                C=C,
                penalty=penalty,
                max_iter=max_iter,
                random_state=random_state,
                solver='liblinear' if penalty == 'l1' else 'lbfgs'
            )
            lr_model.fit(X_train_scaled, y_train)
            
            # Guardar modelos en session state
            st.session_state['dt_model'] = dt_model
            st.session_state['lr_model'] = lr_model
            
            # Calcular métricas
            y_pred_dt = dt_model.predict(X_test)
            y_pred_lr = lr_model.predict(X_test_scaled)
            
            # Reportes de clasificación
            report_dt = classification_report(y_test, y_pred_dt, output_dict=True)
            report_lr = classification_report(y_test, y_pred_lr, output_dict=True)
            
            # Curvas ROC
            fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_model.predict_proba(X_test)[:, 1])
            auc_dt = auc(fpr_dt, tpr_dt)
            
            fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_model.predict_proba(X_test_scaled)[:, 1])
            auc_lr = auc(fpr_lr, tpr_lr)
            
            # Guardar métricas
            metrics_report = f"""
            ## Comparación de Modelos
            
            ### Árbol de Decisión
            - **Accuracy:** {report_dt['accuracy']:.2f}
            - **Precision (Promedio):** {report_dt['macro avg']['precision']:.2f}
            - **Recall (Promedio):** {report_dt['macro avg']['recall']:.2f}
            - **F1-Score (Promedio):** {report_dt['macro avg']['f1-score']:.2f}
            - **AUC:** {auc_dt:.2f}
            
            ### Regresión Logística
            - **Accuracy:** {report_lr['accuracy']:.2f}
            - **Precision (Promedio):** {report_lr['macro avg']['precision']:.2f}
            - **Recall (Promedio):** {report_lr['macro avg']['recall']:.2f}
            - **F1-Score (Promedio):** {report_lr['macro avg']['f1-score']:.2f}
            - **AUC:** {auc_lr:.2f}
            """
            
            st.session_state['metrics_report'] = metrics_report
            save_model(dt_model, MODEL_DT_PATH)
            save_model((lr_model, scaler), MODEL_LR_PATH)
            save_metrics(metrics_report)
            
            st.success("¡Modelos entrenados exitosamente!")
            st.balloons()

# Página de Métricas
elif page == "📈 Métricas":
    st.header("📈 Evaluación de Modelos")
    
    if 'metrics_report' not in st.session_state:
        st.warning("Por favor, entrene los modelos en la página anterior primero.")
        st.stop()
    
    # Mostrar reporte de métricas
    st.markdown(st.session_state['metrics_report'])
    
    # Visualización adicional
    st.subheader("Visualización de Métricas")
    
    # Cargar datos necesarios
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    X_test_scaled = st.session_state['X_test_scaled']
    dt_model = st.session_state['dt_model']
    lr_model = st.session_state['lr_model']
    le = st.session_state['label_encoder']
    
    # Matrices de confusión
    st.write("### Matrices de Confusión")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Árbol de Decisión**")
        y_pred_dt = dt_model.predict(X_test)
        cm_dt = confusion_matrix(y_test, y_pred_dt)
        
        fig, ax = plt.subplots()
        sns.heatmap(
            cm_dt, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=ax
        )
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de Confusión - Árbol de Decisión')
        st.pyplot(fig)
        
        # Métricas detalladas
        st.write("**Métricas por clase:**")
        st.table(pd.DataFrame(classification_report(y_test, y_pred_dt, output_dict=True, target_names=le.classes_)).transpose())
    
    with col2:
        st.write("**Regresión Logística**")
        y_pred_lr = lr_model.predict(X_test_scaled)
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        
        fig, ax = plt.subplots()
        sns.heatmap(
            cm_lr, 
            annot=True, 
            fmt='d', 
            cmap='Greens',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=ax
        )
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de Confusión - Regresión Logística')
        st.pyplot(fig)
        
        # Métricas detalladas
        st.write("**Métricas por clase:**")
        st.table(pd.DataFrame(classification_report(y_test, y_pred_lr, output_dict=True, target_names=le.classes_)).transpose())
    
    # Curvas ROC
    st.write("### Curvas ROC Comparativas")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ROC para Árbol de Decisión
    RocCurveDisplay.from_estimator(
        dt_model, 
        X_test, 
        y_test, 
        ax=ax, 
        name='Árbol de Decisión',
        color='blue'
    )
    
    # ROC para Regresión Logística
    RocCurveDisplay.from_estimator(
        lr_model, 
        X_test_scaled, 
        y_test, 
        ax=ax, 
        name='Regresión Logística',
        color='green'
    )
    
    # Línea de referencia
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Aleatorio', alpha=.8)
    ax.set_title('Curvas ROC Comparativas')
    ax.legend(loc="lower right")
    
    st.pyplot(fig)
    
    # Interpretación de métricas
    st.subheader("Interpretación de Métricas")
    
    st.markdown("""
    - **Accuracy (Exactitud):** Porcentaje de predicciones correctas. Útil cuando las clases están balanceadas.
    - **Precision (Precisión):** De los predichos como positivos, cuántos realmente lo son. Importante cuando los falsos positivos son costosos.
    - **Recall (Sensibilidad):** De los reales positivos, cuántos fueron correctamente identificados. Importante cuando los falsos negativos son costosos.
    - **F1-Score:** Media armónica entre precisión y recall. Buen balance entre ambas métricas.
    - **AUC-ROC:** Mide la capacidad del modelo para distinguir entre clases. 1 = perfecto, 0.5 = aleatorio.
    """)
    
    # Comparación de modelos
    st.subheader("Comparación de Modelos")
    
    st.markdown("""
    **Árbol de Decisión:**
    - Ventajas:
      - Fácil de interpretar y visualizar
      - No requiere escalado de características
      - Maneja bien relaciones no lineales
    - Desventajas:
      - Propenso a overfitting si no se regulariza
      - Pequeños cambios en datos pueden producir árboles muy diferentes
    
    **Regresión Logística:**
    - Ventajas:
      - Resultados interpretables (probabilidades)
      - Eficiente computacionalmente
      - Funciona bien cuando la relación es aproximadamente lineal
    - Desventajas:
      - Requiere escalado de características
      - No captura relaciones complejas sin feature engineering
    
    **Recomendación:**
    - Si la interpretabilidad es clave y las relaciones son no lineales → Árbol de Decisión
    - Si necesitas probabilidades y las relaciones son lineales → Regresión Logística
    """)

# Página de Predicción
elif page == "🔮 Predicción":
    st.header("🔮 Predicción de Nuevos Atletas")
    
    if 'dt_model' not in st.session_state or 'lr_model' not in st.session_state:
        st.warning("Por favor, entrene los modelos en la página de Modelado primero.")
        st.stop()
    
    # Cargar modelos y datos necesarios
    dt_model = st.session_state['dt_model']
    lr_model, scaler = st.session_state['lr_model'], st.session_state['scaler']
    le = st.session_state['label_encoder']
    
    # Selección de modelo
    st.subheader("Selección de Modelo")
    model_choice = st.radio(
        "Seleccione el modelo para realizar predicciones:",
        ["Árbol de Decisión", "Regresión Logística"],
        horizontal=True
    )
    
    # Entradas para predicción
    st.subheader("Datos del Atleta")
    
    col1, col2 = st.columns(2)
    
    with col1:
        frecuencia_cardiaca = st.number_input(
            "Frecuencia Cardíaca Basal (lpm)", 
            min_value=30.0, 
            max_value=120.0, 
            value=72.0,
            step=0.5
        )
        fibras_lentas = st.number_input(
            "Porcentaje de Fibras Lentas (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=50.0,
            step=0.5
        )
    
    with col2:
        fibras_rapidas = st.number_input(
            "Porcentaje de Fibras Rápidas (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=50.0,
            step=0.5
        )
        imc = st.number_input(
            "Índice de Masa Corporal (IMC)", 
            min_value=15.0, 
            max_value=40.0, 
            value=22.0,
            step=0.1
        )
    
    # Crear dataframe con los datos de entrada
    input_data = pd.DataFrame([[
        frecuencia_cardiaca, 
        fibras_lentas, 
        fibras_rapidas, 
        imc
    ]], columns=[
        "frecuencia_cardiaca_basal", 
        "porcentaje_fibras_lentas", 
        "porcentaje_fibras_rapidas", 
        "indice_masa_corporal"
    ])
    
    if st.button("Predecir Tipo de Atleta"):
        with st.spinner("Realizando predicción..."):
            if model_choice == "Árbol de Decisión":
                prediction = dt_model.predict(input_data)
                probabilities = dt_model.predict_proba(input_data)[0]
            else:
                input_data_scaled = scaler.transform(input_data)
                prediction = lr_model.predict(input_data_scaled)
                probabilities = lr_model.predict_proba(input_data_scaled)[0]
            
            # Decodificar predicción
            predicted_class = le.inverse_transform(prediction)[0]
            
            # Mostrar resultados
            st.success(f"**Predicción:** El atleta es **{predicted_class}**")
            
            # Mostrar probabilidades
            st.subheader("Probabilidades Estimadas")
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(
                le.classes_, 
                probabilities, 
                color=['skyblue', 'salmon']
            )
            
            # Añadir etiquetas a las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height, 
                    f'{height:.1%}',
                    ha='center', 
                    va='bottom'
                )
            
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probabilidad")
            ax.set_title("Probabilidad de pertenecer a cada clase")
            st.pyplot(fig)
            
            # Interpretación de características (solo para árbol de decisión)
            if model_choice == "Árbol de Decisión":
                st.subheader("Interpretación de la Decisión")
                
                # Obtener la ruta de decisión
                feature_names = input_data.columns
                decision_path = dt_model.decision_path(input_data)
                
                # Obtener los nodos de la ruta de decisión
                node_indicator = dt_model.decision_path(input_data)
                leaf_id = dt_model.apply(input_data)
                
                # Obtener las reglas de decisión
                st.write("**Reglas de decisión aplicadas:**")
                
                # Mapeo de características
                feature = dt_model.tree_.feature
                threshold = dt_model.tree_.threshold
                
                node_index = node_indicator.indices[node_indicator.indptr[0]:
                                                  node_indicator.indptr[1]]
                
                rules = []
                for node_id in node_index:
                    if leaf_id[0] == node_id:
                        continue
                    
                    if input_data.iloc[0, feature[node_id]] <= threshold[node_id]:
                        threshold_sign = "<="
                    else:
                        threshold_sign = ">"
                    
                    rule = (
                        f"{feature_names[feature[node_id]]} "
                        f"{threshold_sign} {threshold[node_id]:.2f}"
                    )
                    rules.append(rule)
                
                for i, rule in enumerate(rules, 1):
                    st.write(f"{i}. {rule}")
                
                st.info("""
                **Interpretación:** Estas son las condiciones que llevaron al modelo a 
                clasificar al atleta en la categoría predicha.
                """)
            
            # Explicación para regresión logística
            else:
                st.subheader("Contribución de Características")
                
                # Obtener coeficientes
                coefficients = lr_model.coef_[0]
                feature_importance = pd.DataFrame({
                    'Característica': input_data.columns,
                    'Coeficiente': coefficients,
                    'Magnitud': abs(coefficients)
                }).sort_values('Magnitud', ascending=False)
                
                # Mostrar importancia de características
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(
                    data=feature_importance,
                    x='Coeficiente',
                    y='Característica',
                    palette='viridis',
                    ax=ax
                )
                ax.set_title("Contribución de Características en la Regresión Logística")
                ax.set_xlabel("Coeficiente (magnitud indica importancia)")
                st.pyplot(fig)
                
                st.info("""
                **Interpretación:**
                - Coeficientes positivos aumentan la probabilidad de ser velocista.
                - Coeficientes negativos aumentan la probabilidad de ser fondista.
                - La magnitud indica la importancia relativa de cada característica.
                """)