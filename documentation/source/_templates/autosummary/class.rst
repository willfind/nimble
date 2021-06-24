{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree:
      :recursive:
   {% for item in attributes %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
      :recursive:
   {% for item in methods %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
