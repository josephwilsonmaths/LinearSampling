import LinearSampling.LossFunction


def Posterior(network, glm_type = 'DNN', task = 'regression', precision='single', cross_entropy = False, feature_extractor=None, num_features=None, num_outputs=None):
    if glm_type == 'DNN' and task == 'regression':
        posterior = LinearSampling.LossFunction.LeastSquaresRegressionPosterior('ntk', network, precision)

    elif glm_type == 'DNN' and task == 'classification' and not cross_entropy:    
        posterior = LinearSampling.LossFunction.LeastSquaresLogitsPosterior('ntk', network, precision)

    elif glm_type == 'LL' and task == 'regression':
        posterior = LinearSampling.LossFunction.LeastSquaresRegressionPosterior('ck', network, precision)

    elif glm_type == 'LL' and task == 'classification' and not cross_entropy:
        posterior = LinearSampling.LossFunction.LeastSquaresLogitsPosterior('ck', network, precision, feature_extractor, num_features, num_outputs)    
    

    elif glm_type == 'DNN' and task == 'classification' and cross_entropy:
        posterior = LinearSampling.LossFunction.CrossEntropyPosterior('ntk', network, precision)

    elif glm_type == 'LL' and task == 'classification' and cross_entropy:
        posterior = LinearSampling.LossFunction.CrossEntropyPosterior('ck', network, precision, feature_extractor, num_features, num_outputs)

    else:
        raise ValueError('Invalid posterior. GLM options are "DNN" or "LL", task options are "regression" or "classification".')
    
    posterior.set_methodname(glm_type)
    return posterior

            