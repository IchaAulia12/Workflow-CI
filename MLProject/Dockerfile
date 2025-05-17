# Gunakan Python image sebagai base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Salin semua file dari folder MLProject ke /app di dalam container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan script ketika container dimulai
CMD ["python", "modelling.py"]
