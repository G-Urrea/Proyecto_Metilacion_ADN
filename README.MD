# Instrucciones
## Preparativos
Las intrucciones asumen que previamente se han instalado **Anaconda** y **Jupyter Notebook**. 

### Ambiente de desarrollo
Para el uso correcto del proyecto primero debe crearse un ambiente virtual de python, con el objetivo de evitar problemas de dependencias de librerías. El siguiente comando, ejecutado en una terminal, creará un ambiente virtual llamado "meth-env":

`conda create --name meth-env python`

El siguiente comando activará el ambiente virtual creado:

`conda activate meth-env`

A continuación, en la carpeta raíz del proyecto, debe ejecutarse el siguiente comando para instalar las dependencias necesarias:

`pip install -r src/requirements.txt`

Realizando esto ya debería ser posible trabajar con los notebooks sin problemas de dependencias.
### Utilizar Jupyter
Para utilizar el servidor de jupyter notebook de anaconda bastaría utilizar el siguiente comando desde la terminal con el ambiente activado:

`jupyter notebook`

**NOTA**: También es posible seleccionar el kernel manualmente si se inicia jupyter notebook desde una terminal con el ambiente base de conda activado (ambiente que se inicializa con `conda activate base`). Para ello puede que sea necesario ejecutar el siguiente comando:

`conda install nb_conda_kernels`

Alternativamente, se puede utilizar cualquier otra alternativa de preferencia, como *Visual Studio Code* con las extensiones de jupyter y utilizando el kernel correspondiente al ambiente. 
