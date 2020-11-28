class FastTextUtils():

    def __init__(self, model, k=-1):
        self.model = model
        self.k = k

    def get_prediction(self, text):
        prediction = self.model.predict(text, k=self.k)
        class_probas = dict(zip(prediction[0], prediction[1]))
        class_probas = {int(_k.replace('__label__', '')): _v for _k, _v in class_probas.items()}
        return class_probas

    @staticmethod
    def get_model_datasets(data_df, text_col, target_col=''):

        data = ''
        target_val = 1
        for idx, row in data_df.iterrows():
            if target_col != '': target_val = row[target_col]

            row_data = '__label__{} {}'.format(target_val, row[text_col])
            data += row_data + '\n'
        return data

    @staticmethod
    def read_ft_predict_file(fn):
        data = FasttextUtils.read_model_data(fn)
        data_rows = data.split('\n')
        cleaned_rows = [val.replace('__label__', '') for val in data_rows]
        cleaned_rows = [int(val) for val in cleaned_rows if val != '']
        return cleaned_rows

    @staticmethod
    def read_ft_test_file(fn):
        data = FasttextUtils.read_model_data(fn)
        data_rows = data.split('\n')
        cleaned_rows = [val.split(' ')[0].replace('__label__', '') for val in data_rows]
        cleaned_rows = [int(val) for val in cleaned_rows if val != '']
        return cleaned_rows

    @staticmethod
    def read_model_data(fn, encoding='utf-8'):

        data = ''
        try:
            with open(fn, 'r', encoding=encoding) as fobj:
                data = fobj.read()
        except Exception as e:
            logging.error("Failed reading file: %s", fn)
            print(e)
        return data

    @staticmethod
    def write_model_data(data, fn, encoding='utf-8'):
        with open(fn, 'w', encoding=encoding) as fobj:
            fobj.write(data)

    @staticmethod
    def read_ft_predict_prob_file(fn):
        data = FasttextUtils.read_model_data(fn)
        data_rows = data.split('\n')

        class_probas = []
        for data_row in data_rows:
            if data_row.strip() == '': continue
            pred_probas = data_row.split('__label__')
            probas = {}
            for pred_proba in pred_probas:
                if pred_proba == '': continue
                parts = pred_proba.split(' ')
                print(parts)
                probas[int(parts[0].strip())] = float(parts[1].strip())
            class_probas.append(probas)
        proba_df = pd.DataFrame(class_probas)
        return proba_df