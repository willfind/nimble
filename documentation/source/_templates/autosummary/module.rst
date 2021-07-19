{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

.. autosummary::
   :toctree:
   :recursive:

   {%- if attributes %}
   {% for item in attributes %}
   {{ item }}
   {%- endfor %}
   {% endif %}

   {%- if functions %}
   {% for item in functions %}
   {{ item }}
   {%- endfor %}
   {% endif %}

   {%- if classes %}
   {% for item in classes %}
   {{ item }}
   {%- endfor %}
   {% endif %}

   {%- if exceptions %}
   {% for item in exceptions %}
   {{ item }}
   {%- endfor %}
   {% endif %}
