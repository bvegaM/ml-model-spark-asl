from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession

from typing import Union

tags_metadata = [
    {
        "name": "recommender-system",
        "description": "Operations with recommender system. We found **predict** endpoint.",
    }
]

app = FastAPI(openapi_tags=tags_metadata)

# Create session in Spark
spark = SparkSession.builder \
    .appName('recommendationAPI') \
    .getOrCreate()

# Upload ALS model
model = ALSModel.load('./model/als.model')


@app.get('/predict', tags=["recommender-system"],
    summary="Endpoint para obtener recomendaciones de usuarios",
    response_description="Respuesta exitosa con recomendaciones",
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
    })
async def predict(user_id: int, num_recommendations: int):
    """
        Devuelve las recomendaciones para un usuario dado su ID.

        :param user_id: El ID del usuario.

        :param num_recommendations: El número de recomendaciones a devolver.

        :return: Un objeto JSON con la lista de recomendaciones.
        """
    try:
        user_df = spark.createDataFrame([(user_id,) for user_id in [user_id]], ['userId'])

        recommendations = model.recommendForUserSubset(user_df, num_recommendations)

        response = {'recommendations': []}
        if(recommendations.collect()):
            for row in recommendations.collect():
                user_id = row['userId']
                items = [item for item, rating in row['recommendations']]
                response['recommendations'].append({'userId': user_id, 'items': items})
            return response, 200
        else:
            return {"error": 'Usuario no existente'}, 200
    except Exception as e:
        return {"error": str(e)}, 400


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Sistema de recomendación API",
        version="1.0-RELEASE",
        description="Esta es un API de un sistema de recomendación utilizando FAST API y Apache Spark",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
