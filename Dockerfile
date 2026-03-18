# 1. Imagen base oficial de Python
FROM python:3.9-slim

# 2. Establecer variables de entorno para evitar archivos .pyc y asegurar logs fluidos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Instalar dependencias del sistema necesarias para bibliotecas numéricas
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 5. Copiar e instalar dependencias de Python
# Nota: Crearemos el requirements.txt en el siguiente paso
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copiar el resto del código del proyecto
COPY . .

# 7. Exponer el puerto para la interfaz de MLflow (Tracker)
EXPOSE 5000

# 8. Comando por defecto (puede ser sobrescrito en el pipeline de Jenkins)
CMD ["python", "src/main.py"]