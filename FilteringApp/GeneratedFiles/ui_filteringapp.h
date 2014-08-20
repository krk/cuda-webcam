/********************************************************************************
** Form generated from reading UI file 'filteringapp.ui'
**
** Created by: Qt User Interface Compiler version 5.3.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FILTERINGAPP_H
#define UI_FILTERINGAPP_H

#include <GLDualCamView.h>
#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_FilteringAppClass
{
public:
    QAction *actionCapture;
    QAction *actionProcess;
    QWidget *centralWidget;
    QGroupBox *groupBox;
    QListView *lvFilters;
    QWidget *layoutWidget;
    QVBoxLayout *verticalLayout_2;
    QSpacerItem *verticalSpacer;
    QVBoxLayout *verticalLayout;
    QPushButton *pbMoveFilterUp;
    QPushButton *pbRemove;
    QPushButton *pbMoveFilterDown;
    QSpacerItem *verticalSpacer_2;
    QWidget *layoutWidget1;
    QHBoxLayout *horizontalLayout;
    QLabel *lblFilterType;
    QComboBox *cmbFilterType;
    QPushButton *pbAddFilter;
    QFrame *frame;
    GLDualCamView *camDual;
    QGroupBox *grpParameters;
    QWidget *formLayoutWidget;
    QFormLayout *formLayout;
    QLabel *lblParameterCaption;
    QSpinBox *spinParameter;
    QSpacerItem *verticalSpacer_3;
    QSpacerItem *verticalSpacer_4;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *FilteringAppClass)
    {
        if (FilteringAppClass->objectName().isEmpty())
            FilteringAppClass->setObjectName(QStringLiteral("FilteringAppClass"));
        FilteringAppClass->resize(1079, 664);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(FilteringAppClass->sizePolicy().hasHeightForWidth());
        FilteringAppClass->setSizePolicy(sizePolicy);
        FilteringAppClass->setWindowOpacity(1);
        actionCapture = new QAction(FilteringAppClass);
        actionCapture->setObjectName(QStringLiteral("actionCapture"));
        actionCapture->setCheckable(true);
        QIcon icon;
        icon.addFile(QStringLiteral(":/FilteringApp/Resources/picture.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCapture->setIcon(icon);
        actionProcess = new QAction(FilteringAppClass);
        actionProcess->setObjectName(QStringLiteral("actionProcess"));
        actionProcess->setCheckable(true);
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/FilteringApp/Resources/3x3_grid_2.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionProcess->setIcon(icon1);
        centralWidget = new QWidget(FilteringAppClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(10, 60, 391, 241));
        lvFilters = new QListView(groupBox);
        lvFilters->setObjectName(QStringLiteral("lvFilters"));
        lvFilters->setGeometry(QRect(10, 20, 256, 211));
        layoutWidget = new QWidget(groupBox);
        layoutWidget->setObjectName(QStringLiteral("layoutWidget"));
        layoutWidget->setGeometry(QRect(280, 20, 97, 211));
        verticalLayout_2 = new QVBoxLayout(layoutWidget);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        pbMoveFilterUp = new QPushButton(layoutWidget);
        pbMoveFilterUp->setObjectName(QStringLiteral("pbMoveFilterUp"));
        pbMoveFilterUp->setDefault(false);
        pbMoveFilterUp->setFlat(false);

        verticalLayout->addWidget(pbMoveFilterUp);

        pbRemove = new QPushButton(layoutWidget);
        pbRemove->setObjectName(QStringLiteral("pbRemove"));

        verticalLayout->addWidget(pbRemove);

        pbMoveFilterDown = new QPushButton(layoutWidget);
        pbMoveFilterDown->setObjectName(QStringLiteral("pbMoveFilterDown"));
        pbMoveFilterDown->setDefault(false);
        pbMoveFilterDown->setFlat(false);

        verticalLayout->addWidget(pbMoveFilterDown);


        verticalLayout_2->addLayout(verticalLayout);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer_2);

        layoutWidget1 = new QWidget(centralWidget);
        layoutWidget1->setObjectName(QStringLiteral("layoutWidget1"));
        layoutWidget1->setGeometry(QRect(10, 10, 474, 30));
        horizontalLayout = new QHBoxLayout(layoutWidget1);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        lblFilterType = new QLabel(layoutWidget1);
        lblFilterType->setObjectName(QStringLiteral("lblFilterType"));

        horizontalLayout->addWidget(lblFilterType);

        cmbFilterType = new QComboBox(layoutWidget1);
        cmbFilterType->setObjectName(QStringLiteral("cmbFilterType"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(cmbFilterType->sizePolicy().hasHeightForWidth());
        cmbFilterType->setSizePolicy(sizePolicy1);
        cmbFilterType->setMinimumSize(QSize(300, 0));

        horizontalLayout->addWidget(cmbFilterType);

        pbAddFilter = new QPushButton(layoutWidget1);
        pbAddFilter->setObjectName(QStringLiteral("pbAddFilter"));

        horizontalLayout->addWidget(pbAddFilter);

        frame = new QFrame(centralWidget);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setGeometry(QRect(410, 70, 661, 501));
        frame->setStyleSheet(QStringLiteral("background-color: blue;"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        camDual = new GLDualCamView(frame);
        camDual->setObjectName(QStringLiteral("camDual"));
        camDual->setGeometry(QRect(10, 10, 640, 480));
        grpParameters = new QGroupBox(centralWidget);
        grpParameters->setObjectName(QStringLiteral("grpParameters"));
        grpParameters->setGeometry(QRect(10, 310, 391, 80));
        grpParameters->setStyleSheet(QStringLiteral("display: none;"));
        formLayoutWidget = new QWidget(grpParameters);
        formLayoutWidget->setObjectName(QStringLiteral("formLayoutWidget"));
        formLayoutWidget->setGeometry(QRect(10, 20, 371, 51));
        formLayout = new QFormLayout(formLayoutWidget);
        formLayout->setSpacing(6);
        formLayout->setContentsMargins(11, 11, 11, 11);
        formLayout->setObjectName(QStringLiteral("formLayout"));
        formLayout->setContentsMargins(0, 0, 0, 0);
        lblParameterCaption = new QLabel(formLayoutWidget);
        lblParameterCaption->setObjectName(QStringLiteral("lblParameterCaption"));

        formLayout->setWidget(1, QFormLayout::LabelRole, lblParameterCaption);

        spinParameter = new QSpinBox(formLayoutWidget);
        spinParameter->setObjectName(QStringLiteral("spinParameter"));
        spinParameter->setMaximum(255);
        spinParameter->setValue(100);

        formLayout->setWidget(1, QFormLayout::FieldRole, spinParameter);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        formLayout->setItem(2, QFormLayout::FieldRole, verticalSpacer_3);

        verticalSpacer_4 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        formLayout->setItem(0, QFormLayout::FieldRole, verticalSpacer_4);

        FilteringAppClass->setCentralWidget(centralWidget);
        mainToolBar = new QToolBar(FilteringAppClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        mainToolBar->setMovable(false);
        mainToolBar->setIconSize(QSize(30, 30));
        mainToolBar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        mainToolBar->setFloatable(false);
        FilteringAppClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(FilteringAppClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        statusBar->setStyleSheet(QStringLiteral("color: blue; font: 12px;"));
        FilteringAppClass->setStatusBar(statusBar);
#ifndef QT_NO_SHORTCUT
        lblFilterType->setBuddy(cmbFilterType);
#endif // QT_NO_SHORTCUT

        mainToolBar->addSeparator();
        mainToolBar->addAction(actionCapture);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actionProcess);

        retranslateUi(FilteringAppClass);

        QMetaObject::connectSlotsByName(FilteringAppClass);
    } // setupUi

    void retranslateUi(QMainWindow *FilteringAppClass)
    {
        FilteringAppClass->setWindowTitle(QApplication::translate("FilteringAppClass", "OpenCV + CUDA + Qt + C++AMP", 0));
        actionCapture->setText(QApplication::translate("FilteringAppClass", "Capture", 0));
#ifndef QT_NO_TOOLTIP
        actionCapture->setToolTip(QApplication::translate("FilteringAppClass", "Starts capturing.", 0));
#endif // QT_NO_TOOLTIP
        actionProcess->setText(QApplication::translate("FilteringAppClass", "Process", 0));
#ifndef QT_NO_TOOLTIP
        actionProcess->setToolTip(QApplication::translate("FilteringAppClass", "Starts processing the captured images.", 0));
#endif // QT_NO_TOOLTIP
        groupBox->setTitle(QApplication::translate("FilteringAppClass", "Filter Chain", 0));
        pbMoveFilterUp->setText(QApplication::translate("FilteringAppClass", "Move Up", 0));
        pbRemove->setText(QApplication::translate("FilteringAppClass", "Remove", 0));
        pbMoveFilterDown->setText(QApplication::translate("FilteringAppClass", "Move Down", 0));
#ifndef QT_NO_TOOLTIP
        lblFilterType->setToolTip(QString());
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        lblFilterType->setStatusTip(QApplication::translate("FilteringAppClass", "Select a filter type and press add filter to add filter to the chain.", 0));
#endif // QT_NO_STATUSTIP
        lblFilterType->setText(QApplication::translate("FilteringAppClass", "Filte&r Type:", 0));
        pbAddFilter->setText(QApplication::translate("FilteringAppClass", "Add Filter", 0));
        grpParameters->setTitle(QApplication::translate("FilteringAppClass", "Selected Filter Parameters", 0));
        lblParameterCaption->setText(QApplication::translate("FilteringAppClass", "Caption:", 0));
        spinParameter->setPrefix(QString());
    } // retranslateUi

};

namespace Ui {
    class FilteringAppClass: public Ui_FilteringAppClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FILTERINGAPP_H
