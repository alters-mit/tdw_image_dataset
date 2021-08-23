from py_md_doc import PyMdDoc


md = PyMdDoc(input_directory="tdw_image_dataset",
             files=["image_dataset.py",
                    "image_position.py"])
md.get_docs(output_directory="doc")
