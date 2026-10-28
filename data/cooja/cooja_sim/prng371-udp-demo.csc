<?xml version="1.0" encoding="UTF-8"?>
<simconf>
  <simulation>
    <title>PRNG-371 UDP Image Encryption Demo (2 Nodes)</title>
    <randomseed>123456</randomseed>
    <motedelay_us>1000000</motedelay_us>
    <radiomedium>
      org.contikios.cooja.radiomediums.UDGM
      <transmitting_range>50.0</transmitting_range>
      <interference_range>100.0</interference_range>
      <!-- Packet loss simulation: 85% success rate = 15% packet loss -->
      <success_ratio_tx>0.85</success_ratio_tx>
      <success_ratio_rx>1.0</success_ratio_rx>
    </radiomedium>
    <events>
      <logoutput>40000</logoutput>
    </events>

    <!-- Sensor Mote Type -->
    <motetype>
      org.contikios.cooja.contikimote.ContikiMoteType
      <identifier>sensor</identifier>
      <description>PRNG-371 Sensor Node</description>
      <source>[CONFIG_DIR]/sensor-node.c</source>
      <commands>$(MAKE) -j$(CPUS) sensor-node.cooja TARGET=cooja</commands>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Battery</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiVib</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiMoteID</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRS232</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiBeeper</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.RimeAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiIPAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRadio</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiButton</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiPIR</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiClock</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiLED</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiCFS</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiEEPROM</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Mote2MoteRelations</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.MoteAttributes</moteinterface>
    </motetype>

    <!-- Gateway Mote Type -->
    <motetype>
      org.contikios.cooja.contikimote.ContikiMoteType
      <identifier>gateway</identifier>
      <description>PRNG-371 Gateway Node</description>
      <source>[CONFIG_DIR]/gateway-node.c</source>
      <commands>$(MAKE) -j$(CPUS) gateway-node.cooja TARGET=cooja</commands>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Battery</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiVib</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiMoteID</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRS232</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiBeeper</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.RimeAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiIPAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRadio</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiButton</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiPIR</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiClock</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiLED</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiCFS</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiEEPROM</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Mote2MoteRelations</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.MoteAttributes</moteinterface>
    </motetype>
    <!-- Relay Mote Type -->
    <motetype>
      org.contikios.cooja.contikimote.ContikiMoteType
      <identifier>relay</identifier>
      <description>Relay Node (image part exporter)</description>
      <source>[CONFIG_DIR]/relay-node.c</source>
      <commands>$(MAKE) -j$(CPUS) relay-node.cooja TARGET=cooja</commands>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Battery</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiVib</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiMoteID</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRS232</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiBeeper</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.RimeAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiIPAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRadio</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiButton</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiPIR</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiClock</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiLED</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiCFS</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiEEPROM</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Mote2MoteRelations</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.MoteAttributes</moteinterface>
    </motetype>


    <!-- Sensor Node (Mote 1) -->
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>20.0</x>
        <y>20.0</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>1</id>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiRadio
        <bitrate>250.0</bitrate>
      </interface_config>
      <motetype_identifier>sensor</motetype_identifier>
    </mote>

    <!-- Gateway Node (Mote 2) -->
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>40.0</x>
        <y>20.0</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>2</id>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiRadio
        <bitrate>250.0</bitrate>
      </interface_config>
      <motetype_identifier>gateway</motetype_identifier>
    </mote>
    <!-- Relay Node 3 -->
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>35.0</x>
        <y>20.0</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>3</id>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiRadio
        <bitrate>250.0</bitrate>
      </interface_config>
      <motetype_identifier>relay</motetype_identifier>
    </mote>

    <!-- Relay Node 4 -->
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>45.0</x>
        <y>20.0</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>4</id>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiRadio
        <bitrate>250.0</bitrate>
      </interface_config>
      <motetype_identifier>relay</motetype_identifier>
    </mote>

    <!-- Relay Node 5 -->
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>40.0</x>
        <y>15.0</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>5</id>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiRadio
        <bitrate>250.0</bitrate>
      </interface_config>
      <motetype_identifier>relay</motetype_identifier>
    </mote>

    <!-- Relay Node 6 -->
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>40.0</x>
        <y>25.0</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>6</id>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiRadio
        <bitrate>250.0</bitrate>
      </interface_config>
      <motetype_identifier>relay</motetype_identifier>
    </mote>

    <!-- Relay Node 7 -->
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>35.0</x>
        <y>15.0</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>7</id>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiRadio
        <bitrate>250.0</bitrate>
      </interface_config>
      <motetype_identifier>relay</motetype_identifier>
    </mote>

    <!-- Relay Node 8 -->
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>45.0</x>
        <y>15.0</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>8</id>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiRadio
        <bitrate>250.0</bitrate>
      </interface_config>
      <motetype_identifier>relay</motetype_identifier>
    </mote>

    <!-- Relay Node 9 -->
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>35.0</x>
        <y>25.0</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>9</id>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiRadio
        <bitrate>250.0</bitrate>
      </interface_config>
      <motetype_identifier>relay</motetype_identifier>
    </mote>

    <!-- Relay Node 10 -->
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>45.0</x>
        <y>25.0</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>10</id>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiRadio
        <bitrate>250.0</bitrate>
      </interface_config>
      <motetype_identifier>relay</motetype_identifier>
    </mote>


  </simulation>

  <plugin>
    org.contikios.cooja.plugins.Visualizer
    <plugin_config>
      <mote_relations>true</mote_relations>
      <skin>org.contikios.cooja.plugins.skins.IDVisualizerSkin</skin>
      <skin>org.contikios.cooja.plugins.skins.GridVisualizerSkin</skin>
      <skin>org.contikios.cooja.plugins.skins.TrafficVisualizerSkin</skin>
      <skin>org.contikios.cooja.plugins.skins.UDGMVisualizerSkin</skin>
      <viewport>3.0 0.0 0.0 3.0 0.0 0.0</viewport>
    </plugin_config>
    <width>400</width>
    <z>1</z>
    <height>400</height>
    <location_x>1</location_x>
    <location_y>1</location_y>
  </plugin>

  <plugin>
    org.contikios.cooja.plugins.LogListener
    <plugin_config>
      <filter />
      <formatted_time />
      <coloring />
    </plugin_config>
    <width>1000</width>
    <z>2</z>
    <height>300</height>
    <location_x>1</location_x>
    <location_y>420</location_y>
  </plugin>

  <plugin>
    org.contikios.cooja.plugins.RadioLogger
    <plugin_config>
      <split>150</split>
      <analyzed>true</analyzed>
      <format>ANALYZED</format>
    </plugin_config>
    <width>500</width>
    <z>3</z>
    <height>300</height>
    <location_x>1</location_x>
    <location_y>730</location_y>
  </plugin>


  <plugin>
    org.contikios.cooja.serialsocket.SerialSocketServer
    <mote_arg>1</mote_arg>
    <plugin_config>
      <port>60000</port>
    </plugin_config>
    <width>180</width>
    <z>4</z>
    <height>80</height>
    <location_x>500</location_x>
    <location_y>100</location_y>
  </plugin>

  <plugin>
    org.contikios.cooja.serialsocket.SerialSocketServer
    <mote_arg>2</mote_arg>
    <plugin_config>
      <port>60001</port>
    </plugin_config>
    <width>180</width>
    <z>4</z>
    <height>80</height>
    <location_x>500</location_x>
    <location_y>100</location_y>
  </plugin>

  <plugin>
    org.contikios.cooja.serialsocket.SerialSocketServer
    <mote_arg>3</mote_arg>
    <plugin_config>
      <port>60002</port>
    </plugin_config>
    <width>180</width>
    <z>5</z>
    <height>80</height>
    <location_x>700</location_x>
    <location_y>100</location_y>
  </plugin>

  <plugin>
    org.contikios.cooja.serialsocket.SerialSocketServer
    <mote_arg>4</mote_arg>
    <plugin_config>
      <port>60003</port>
    </plugin_config>
    <width>180</width>
    <z>6</z>
    <height>80</height>
    <location_x>900</location_x>
    <location_y>100</location_y>
  </plugin>

  <plugin>
    org.contikios.cooja.serialsocket.SerialSocketServer
    <mote_arg>5</mote_arg>
    <plugin_config>
      <port>60004</port>
    </plugin_config>
    <width>180</width>
    <z>7</z>
    <height>80</height>
    <location_x>1100</location_x>
    <location_y>100</location_y>
  </plugin>

  <plugin>
    org.contikios.cooja.serialsocket.SerialSocketServer
    <mote_arg>6</mote_arg>
    <plugin_config>
      <port>60005</port>
    </plugin_config>
    <width>180</width>
    <z>8</z>
    <height>80</height>
    <location_x>500</location_x>
    <location_y>220</location_y>
  </plugin>

  <plugin>
    org.contikios.cooja.serialsocket.SerialSocketServer
    <mote_arg>7</mote_arg>
    <plugin_config>
      <port>60006</port>
    </plugin_config>
    <width>180</width>
    <z>9</z>
    <height>80</height>
    <location_x>700</location_x>
    <location_y>220</location_y>
  </plugin>

  <plugin>
    org.contikios.cooja.serialsocket.SerialSocketServer
    <mote_arg>8</mote_arg>
    <plugin_config>
      <port>60007</port>
    </plugin_config>
    <width>180</width>
    <z>10</z>
    <height>80</height>
    <location_x>900</location_x>
    <location_y>220</location_y>
  </plugin>

  <plugin>
    org.contikios.cooja.serialsocket.SerialSocketServer
    <mote_arg>9</mote_arg>
    <plugin_config>
      <port>60008</port>
    </plugin_config>
    <width>180</width>
    <z>11</z>
    <height>80</height>
    <location_x>1100</location_x>
    <location_y>220</location_y>
  </plugin>

  <plugin>
    org.contikios.cooja.plugins.ScriptRunner
    <plugin_config>
      <scriptfile>[CONFIG_DIR]/simulation.js</scriptfile>
      <active>true</active>
    </plugin_config>
    <width>400</width>
    <z>0</z>
    <height>300</height>
    <location_x>0</location_x>
    <location_y>0</location_y>
  </plugin>
</simconf>
