'''
Funciones utiles para la exploración de datos
'''

import seaborn as sns
import matplotlib.pyplot as plt

def violin_plot(data, x, y, hue, title, x_label='Características', y_label='Valor'):
    '''
    Gráfico de violín estandar, con separación por colores
    '''
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 8))
    sns.violinplot(x=x, y=y, hue=hue, data=data, split=True)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
