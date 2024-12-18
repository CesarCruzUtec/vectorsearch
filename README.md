# Busqueda de imagenes mediante vectorización

## Descripción

Este proyecto tiene como objetivo la busqueda de imagenes mediante la vectorización de las mismas. Para ello se ha utilizado librerías como `torch` y `clip` para la vectorización de las imágenes y `psycopg2` para la conexión con la base de datos.

## Instalación

Las librerías necesarias para la ejecución del proyecto se encuentran en el archivo `requirements.txt`. Para instalarlas, se puede ejecutar el siguiente comando:

```bash
pip install -r requirements.txt
```

## Creación de la base de datos

### Instalación de PostgreSQL y pgvector

```bash
# Instalación de PostgreSQL
sudo pacman -S postgresql

# Inicializar la base de datos
sudo -iu postgres
initdb --locale $LANG -E UTF8 -D '/var/lib/postgres/data'
exit

# Iniciar el servicio
sudo systemctl start postgresql

# Instalación de pgvector
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### Configuración de la base de datos

```bash
# Crear una nueva base de datos y un nuevo usuario
sudo -iu postgres
createuser tu_usuario
createdb -O tu_usuario tu_base_de_datos
psql -d tu_base_de_datos
```

```sql
-- Habilitar la extensión pgvector
CREATE EXTENSION vector;

-- Crear la tabla de vectores
CREATE TABLE imagenes (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(255),
    embedding VECTOR(512)
);

-- Indexar la columna de vectores
CREATE INDEX ON imagenes USING ivfflat (embedding) WITH (lists = 100);
```

### Otorgrar permisos de acceso al usuario

```bash
# Logearse como superusuario
psql -U postgres -d vectorsearch
```

```sql
-- Cambiar propietario de la tabla
ALTER TABLE imagenes OWNER TO yubi;

-- Dar permisos de acceso
GRANT ALL PRIVILEGES ON TABLE imagenes TO yubi;
```
