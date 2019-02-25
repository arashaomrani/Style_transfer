
# coding: utf-8

# In[4]:


from keras.models import model_from_json
import coremltools
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
top_model = model_from_json(loaded_model_json)


# In[5]:


top_model.load_weights('candy_separate_style_content_reflectionpadding_32filter_3res_s9000.h5')


# In[ ]:


coreml_model = coremltools.converters.keras.convert(
    top_model, input_names=['inp'], output_names=['outp'])
coreml_model.save('kerasmodel.mlmodel')

