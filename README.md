# EL_proyecto
lo primero que hice fue crear un repositorio en github
luego de crear el proyecto hice una carpeta llamada clon 
para despues abrir visual studio con la cmd
luego copiamos el codigo a la terminal agregandole el git code
para agregar el gitcode tuve que instalar la aplicacion git y despues instalar python
para seguir con las instrucciones y insertar el codigo recommendation_system.py
luego hice commit y push para poner el codigo git add

## Instrucciones para ejecutar el sistema de recomendación

1. Instala las dependencias necesarias ejecutando en la terminal:
   ```
   pip install numpy pandas scikit-learn
   ```
2. Asegúrate de que el archivo del dataset esté en la ruta `EL_proyecto/dataset/ratings.csv`.
3. Ejecuta el script desde la carpeta `Clon` con el siguiente comando:
   ```
   python EL_proyecto/recommendation_system.py
   ```
4. El sistema mostrará recomendaciones para un usuario del dataset por consola.

Puedes modificar el valor de `user_id` en el archivo `recommendation_system.py` para obtener recomendaciones para otros usuarios.