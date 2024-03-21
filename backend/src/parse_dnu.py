import json
import re


def parse_dnu(text):
    def parse_articulos(articulos_text):
        articulos = {}
        for art in re.finditer(r'(?<!")ARTÍCULO (\d+)\.[–-] (.*?)(?=(?<!")ARTÍCULO \d+\.[–-]|\Z)', articulos_text, re.DOTALL):
            articulo_number = art.group(1).strip()
            articulo_text = art.group(2).strip()
            articulos[f'Articulo {articulo_number}'] = articulo_text
        return articulos

    def parse_capitulos(capitulos_text):
        first_match = True
        span_start = 0
        capitulos = {}
        for chap in re.finditer(r'(?<!")Capítulo (\w+) [–-] (.*?)(?=\n\n|Capítulo \w+ [–-]|\Z)(.*?)(?=Capítulo \w+ [–-]|\Z)', capitulos_text, re.DOTALL):
            if first_match:
                first_match = False
                span_start = chap.span()[0]

                if span_start > 0:
                    articulos = parse_articulos(capitulos_text[:span_start])
                    capitulos.update(articulos)

            capitulo_number = chap.group(1).strip()
            capitulo_title = chap.group(2).strip()
            capitulo_content = chap.group(3).strip()

            capitulos[f'Capitulo {capitulo_number}'] = {
                "title": capitulo_title,
                "contents": parse_articulos(capitulo_content)
            }

        if first_match:
            articulos = parse_articulos(capitulos_text)
            capitulos.update(articulos)

        return capitulos

    document_structure = {}

    for titulo in re.finditer(r'(?<!")Título (\w+) [–-] (.*?)(?=\n\n|Título|\Z)(.*?)(?=Título \w+ [–-]|\Z)', text, re.DOTALL):
        titulo_number = titulo.group(1).strip()
        titulo_title = titulo.group(2).strip()
        titulo_content = titulo.group(3).strip()

        if 'Capítulo' in titulo_content:
            document_structure[f'Título {titulo_number}'] = {
                "title": titulo_title,
                "contents": parse_capitulos(titulo_content)
            }
        else:
            document_structure[f'Título {titulo_number}'] = {
                "title": titulo_title,
                "contents": parse_articulos(titulo_content)
            }

    return document_structure

source_path = r"C:\Users\pablo\Desktop\SandboxAI\LeyDeAlquileres-backend\data\LaLeyDeMilei-raw\decreto.txt"
target_path = r"C:\Users\pablo\Desktop\SandboxAI\LeyDeAlquileres-backend\data\LaLeyDeMilei-raw\decreto.json"

with open(source_path, 'r', encoding='utf-8') as file:
    file_content = file.read()

# Re-parse the sample text with the corrected function
parsed_document_corrected = parse_dnu(file_content)

with open(target_path, 'w', encoding='utf-8') as f:
    json.dump(parsed_document_corrected, f, ensure_ascii=False, indent=4)









sample_text = """
Título I – [T1 title]

ARTÍCULO 1°.- [Art 1 Text]

ARTÍCULO 2°.- [Art 2 Text]

...

Título 2 – [T2 title]

ARTÍCULO 6°.- [Art 6 Text]

ARTÍCULO 7°.- [Art 7 Text]

Capítulo I – [T2 Chap 1 Title]

ARTÍCULO 8°.- [Art 8 Text]

...

Capítulo 2 – [T2 Chap 2 Title]

ARTÍCULO 11°.- [Art 11 Text]

...

Título 3 – [T3 title]

ARTÍCULO 16°.- [Art 16 Text]

ARTÍCULO 17°.- [Art 17 Text]

...

Capítulo I – [T3 Chap 1 Title]

ARTÍCULO 20°.- [Art 20 Text]

...

Capítulo 2 – [T3 Chap 2 Title]

ARTÍCULO 23°.- [Art 23 Text]

...
"""



sample_text_2 = """

Título I – BASES PARA LA RECONSTRUCCIÓN DE LA ECONOMÍA ARGENTINA

ARTÍCULO 1.- EMERGENCIA. Declárase la emergencia pública en materia económica, financiera.

ARTÍCULO 2.- DESREGULACIÓN. El Estado Nacional promoverá y asegurará la vigencia efectiva, en todo el territorio nacional.

Título II – DESREGULACIÓN ECONÓMICA

ARTÍCULO 3.- Derógase la Ley N° 18.425.

ARTÍCULO 4.- Derógase la Ley N° 26.992.

Capítulo I – Banco de la Nación Argentina (Ley N° 21.799)

ARTÍCULO 5.- Derógase el artículo 2° de la Ley N° 21.799.

Capítulo II – Tarjetas de crédito (Ley N° 25.065)

ARTÍCULO 6.- Deróganse los artículos 5°, 7°, 8°, 9°, 17, 32, 35, 53 y 54 de la Ley N° 25.065.

ARTÍCULO 7.- Sustitúyese el artículo 1° de la Ley N° 25.065 por el siguiente:

“ARTÍCULO 1°.- Se entiende por sistema de Tarjeta de Crédito al conjunto de contratos individuales cuya finalidad.

Título III – REFORMA DEL ESTADO

ARTÍCULO 8.- Derógase el Decreto - Ley N° 15.349/46.

ARTÍCULO 9.- Derógase la Ley N° 13.653.

Capítulo I – Reforma del Estado (Ley N° 23.696)

ARTÍCULO 10.- Derógase el tercer párrafo del artículo 9° de la Ley N° 23.696.

ARTÍCULO 11.- Derógase el artículo 29 de la Ley N° 23.696.

Capítulo II - Transformación de empresas del Estado en Sociedades Anónimas

ARTÍCULO 12.- Las sociedades o empresas con participación del Estado.
"""

sample_text_3 = """
ARTÍCULO 170.- Derógase la Ley N° 24.695.

Título VIII – ENERGÍA

ARTÍCULO 171.- Derógase el Decreto N° 1060/00.

ARTÍCULO 172.- Derógase el Decreto N° 1491/02.

ARTÍCULO 173.- Derógase el Decreto N° 634/03.

ARTÍCULO 174.- Derógase la Ley N° 25.822.

ARTÍCULO 175.- Derógase el Decreto N° 311/06.

Capítulo I - Régimen de Fomento a la generación distribuida de energía renovable integrada a la red eléctrica (Ley N° 27.424)

ARTÍCULO 176.- Deróganse los artículos 16 a 37 de la Ley N° 27.424.

ARTÍCULO 177.- Facúltase a la SECRETARÍA DE ENERGÍA del MINISTERIO DE ECONOMÍA a redeterminar la estructura de subsidios vigentes a fin de asegurar a los usuarios finales el acceso al consumo básico y esencial de.

Título IX - AEROCOMERCIAL

ARTÍCULO 178.- Derógase el Decreto - Ley N° 12.507/56.

ARTÍCULO 179.- Derógase la Ley N° 19.030.

ARTÍCULO 180.- Derógase el Decreto N° 1654/02.

Capítulo I - Código Aeronáutico (Ley N° 17.285)

ARTÍCULO 181.- Sustitúyese el artículo 1° de la Ley Nº 17.285 y sus modificatorias por el siguiente:"""
