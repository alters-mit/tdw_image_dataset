from py_md_doc import PyMdDoc


md = PyMdDoc(input_directory="tdw_image_dataset",
             files=["image_dataset.py",
                    "image_position_avatar_object.py"])
md.get_docs(output_directory="doc")
