
import requests


def initializeAnnotation(dataset_name, annotation_name, logger):
    logger.info(f'Initialize annotation {annotation_name} of dataset {dataset_name}')

    annotation = {
        'name': annotation_name,
        'dataset': dataset_name
    }
    request = requests.post('https://feed-uvl.ifi.uni-heidelberg.de/hitec/orchestration/concepts/annotationinit/',
                            json=annotation)

    return request.status_code


def getAnnotation(annotation_name, logger):
    logger.info(f'Get created annotation')
    request = requests.get(
        f'https://feed-uvl.ifi.uni-heidelberg.de/hitec/repository/concepts/annotation/name/{annotation_name}')
    return request.json()


def addCodesToTokens(annotation, codes):
    for code in codes:
        index = code["tokens"][0]
        annotation["tokens"][index]["num_name_codes"] = 1
        annotation["tokens"][index]["num_tore_codes"] = 1
    return annotation


def addClassificationToAnnotation(annotation_name, codes, logger):
    annotation = getAnnotation(annotation_name, logger)
    annotation["codes"] = codes
    return addCodesToTokens(annotation, codes)


def storeAnnotation(annotation, logger):
    logger.info(f'Storing annotation')

    request = requests.post('https://feed-uvl.ifi.uni-heidelberg.de/hitec/repository/concepts/store/annotation/',
                            json=annotation)

    return request.status_code


def createNewAnnotation(dataset_name, annotation_name, codes, logger):
    status_code = initializeAnnotation(dataset_name, annotation_name, logger)
    if status_code == 200:
        annotation = addClassificationToAnnotation(annotation_name, codes, logger)
        storeAnnotation(annotation, logger)
