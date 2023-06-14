from data_utils import dataset_utils
import source.config as config


def qa_specific_dataloading(tokenizer, input_function=None):
    train, eval, test = [], [], []
    for dataset_name in config.using_datasets.split(","):
        data_train, data_eval, data_test = dataset_utils.load_dataset(
            tokenizer, dataset_name=dataset_name, input_function=input_function
        )
        (
            data_train,
            data_eval,
            data_test,
        ) = dataset_utils.get_qa_datasets_from_slot_filling_datasets(
            data_train, data_eval, data_test, input_function=input_function
        )

        train.append(data_train)
        eval.append(data_eval)
        test.append(data_test)
    train_d = dataset_utils.concat_datasets(train)
    # val_d = dataset_utils.concat_datasets(eval)
    # test_d = dataset_utils.concat_datasets(test)
    return train_d, eval[0], test[0]


def qa_specific_dataloading_for_pretraining(tokenizer, input_function=None):
    train, eval, test = [], [], []
    for dataset_name in config.using_datasets.split(","):
        data_train, data_eval, data_test = dataset_utils.load_dataset(
            tokenizer, dataset_name=dataset_name, input_function=input_function
        )
        (
            data_train,
            data_eval,
            data_test,
        ) = dataset_utils.get_qa_datasets_for_pretraining(
            data_train, data_eval, data_test, input_function=input_function
        )
        train.append(data_train)
        eval.append(data_eval)
        test.append(data_test)
    train_d = dataset_utils.concat_datasets(train)
    return train_d, eval[0], test[0]


def slot_filling_dataloading(tokenizer, model_input_function):
    if len(config.using_datasets.split(",")) > 1:
        print(
            "Slot filling currently does not support loading multiple datasets! Using the first one..."
        )
    dataset_name = config.using_datasets.split(",")[0]
    return dataset_utils.load_dataset(
        tokenizer, dataset_name=dataset_name, input_function=model_input_function
    )
