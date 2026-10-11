%global app_id app.voxelink.editor
%global debug_package %{nil}

Name:           voxel-ink
Version:        0.1.0
Release:        1%{?dist}
Summary:        Native volumetric voxel editor
License:        MIT OR Apache-2.0
URL:            https://github.com/kaufmann-dev/voxel-ink
Source0:        voxel-ink
Source1:        %{app_id}.desktop
Source2:        %{app_id}.xml
Source3:        %{app_id}.metainfo.xml
Source4:        %{app_id}.png
ExclusiveArch:  x86_64

BuildRequires:  appstream
BuildRequires:  desktop-file-utils
BuildRequires:  libxml2
Requires:       libX11
Requires:       libXcursor
Requires:       libXi
Requires:       libXrandr
Requires:       libglvnd-egl
Requires:       libglvnd-gles
Requires:       libwayland-client
Requires:       libxkbcommon
Requires:       libxkbcommon-x11
Requires:       vulkan-loader

%description
VoxelInk is a native local-only editor for NIfTI, BINVOX, and single-model
MagicaVoxel volumetric voxel files.

%prep

%build

%install
install -Dpm0755 %{SOURCE0} %{buildroot}%{_bindir}/voxel-ink
install -Dpm0644 %{SOURCE1} %{buildroot}%{_datadir}/applications/%{app_id}.desktop
install -Dpm0644 %{SOURCE2} %{buildroot}%{_datadir}/mime/packages/%{app_id}.xml
install -Dpm0644 %{SOURCE3} %{buildroot}%{_metainfodir}/%{app_id}.metainfo.xml
install -Dpm0644 %{SOURCE4} %{buildroot}%{_datadir}/icons/hicolor/512x512/apps/%{app_id}.png

%check
desktop-file-validate %{SOURCE1}
appstreamcli validate --no-net %{SOURCE3}
xmllint --noout %{SOURCE2} %{SOURCE3}

%files
%{_bindir}/voxel-ink
%{_datadir}/applications/%{app_id}.desktop
%{_datadir}/mime/packages/%{app_id}.xml
%{_metainfodir}/%{app_id}.metainfo.xml
%{_datadir}/icons/hicolor/512x512/apps/%{app_id}.png

%changelog
* Sun Oct 11 2026 VoxelInk maintainers <maintainers@voxelink.invalid> - 0.1.0-1
- Build the first native Linux package
