from setuptools import setup, find_packages

# Чтение зависимостей из requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='tidyhebo',  # Имя вашего пакета
    version='0.1.0',    # Версия пакета
    packages=find_packages(),  # Автоматически находит все пакеты
    extras_require={
        "test": ["pytest"],  # Тестовые зависимости
    },
    install_requires=requirements,  # Зависимости из requirements.txt
    author='Your Name',
    author_email='your.email@example.com',
    description='Описание вашего пакета',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_package',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)