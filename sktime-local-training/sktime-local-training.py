# SKtime RocketClassifier

import sagemaker
from sagemaker.sklearn import SKLearn
from sktime.datasets import load_from_tsfile_to_dataframe

from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer


dummy_role = 'arn:aws:iam::170802245450:role/service-role/AmazonSageMaker-ExecutionRole-20221013T155418'
image_uri = '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3'
def main():
    sklearn = SKLearn(
        entry_point="sktime_train_script_hivecote2.py",
        source_dir='code',
        image_uri=image_uri,
        instance_type="local",
        role = dummy_role
    )

    train_file = "file://./datasets/train.ts"
    test_file = "file://./datasets/test.ts"

    print("Starting model training")
    sklearn.fit({"train": train_file, "test": test_file})
    print("Completed model training")

    print('Deploying endpoint in local mode')
    # predictor = sklearn.deploy(initial_instance_count=1, 
    #                             instance_type='local',
    #                             serializer=NumpySerializer())
                        


    # X_test, y_test = load_from_tsfile_to_dataframe("datasets/test.ts")
    # # X_test_frame = X_test.tail(1).dim_0.to_numpy()

    # # y_predict = predictor.predict(X_test_frame)

    # # print(y_predict)

    # y_predict = []
    # for i,data in X_test.iterrows():
    #     frame = data.to_numpy()
    #     y_predict.append(predictor.predict(frame)[0])

    # print(y_predict)

if __name__ == "__main__":
    main()