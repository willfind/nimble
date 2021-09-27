{{ objname | escape | underline }}

.. currentmodule:: {{ module }}



.. autoclass:: {{ objname }}
   :no-members:

   {% set vars = {'attrs': False, 'methods': False} %}

   {%- block attributes -%}

   {% if attributes %}
   {# only want to add attributes section if public attributes exist #}
   {% for item in attributes %}
     {%- if not item.startswith('_') -%}
       {%- if not vars.attrs -%}
         {% if vars.update({'attrs': True}) %} {%- endif -%}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree:
      :recursive:

        {% endif %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {%- endif %}
   {% endblock %}

   {%- block methods -%}

   {% if methods %}
   {# only want to add methods section if public methods exist #}
   {% for item in methods %}
      {%- if not item.startswith('_') -%}
        {%- if not vars.methods -%}
          {% if vars.update({'methods': True}) %} {%- endif -%}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
      :recursive:

        {% endif %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {%- endif %}
   {% endblock %}
